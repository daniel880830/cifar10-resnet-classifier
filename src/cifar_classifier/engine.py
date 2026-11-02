"""GPU-resident training loop shared by every PyTorch experiment."""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    """Map ``auto`` / ``cpu`` / ``cuda`` to a usable device.

    Raises:
        RuntimeError: If ``cuda`` is requested but unavailable.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed Python-side and PyTorch RNGs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Preprocessor:
    """Converts uint8 NHWC images to normalised float NCHW tensors.

    ``unit`` divides by 255; ``standard`` additionally
    subtracts the per-channel mean and divides by the std of the training images.
    """

    mode: str = "unit"
    mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @classmethod
    def fit(cls, mode: str, images: np.ndarray) -> Preprocessor:
        """Learn normalisation statistics from training images only.

        Raises:
            ValueError: If ``mode`` is unknown.
        """
        if mode == "unit":
            return cls("unit")
        if mode == "standard":
            scaled = images.reshape(-1, 3).astype(np.float64) / 255.0
            return cls("standard", scaled.mean(0).tolist(), scaled.std(0).tolist())
        raise ValueError(f"Unknown normalisation '{mode}'")

    def __call__(self, batch_u8: torch.Tensor) -> torch.Tensor:
        """Normalise a uint8 ``(N, 3, H, W)`` tensor."""
        x = batch_u8.float().div_(255.0)
        if self.mode == "standard":
            mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
        return x

    def state(self) -> dict[str, Any]:
        """Serialisable form stored with the model."""
        return {"mode": self.mode, "mean": self.mean, "std": self.std}


def to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 NHWC numpy → uint8 NCHW tensor on ``device`` (kept uint8 to save memory)."""
    return (
        torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous().to(device)
    )


def augment(x: torch.Tensor, generator: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Random crop (reflect padding) and random horizontal flip, batched on-device."""
    n, _, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (n,), device=x.device, generator=generator)
    dx = torch.randint(0, 2 * pad + 1, (n,), device=x.device, generator=generator)
    rows = (dy[:, None] + torch.arange(h, device=x.device)).view(n, 1, h, 1)
    cols = (dx[:, None] + torch.arange(w, device=x.device)).view(n, 1, 1, w)
    batch = torch.arange(n, device=x.device).view(n, 1, 1, 1)
    channels = torch.arange(x.shape[1], device=x.device).view(1, -1, 1, 1)
    cropped = padded[batch, channels, rows, cols]
    flip = torch.rand(n, device=x.device, generator=generator) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), cropped.flip(3), cropped)


@dataclass
class TrainConfig:
    """Optimisation settings parsed from an experiment."""

    epochs: int
    batch_size: int
    optimizer: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0
    scheduler: str = "none"
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    early_stopping_patience: int | None = None
    label_smoothing: float = 0.0
    augment: bool = False
    amp: bool = False
    seed: int = 0

    @classmethod
    def from_experiment(cls, exp: dict[str, Any]) -> TrainConfig:
        """Pick the known keys out of an experiment dict."""
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in exp.items() if k in keys})


def _optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, eps=1e-7, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    preprocess: Preprocessor,
    images: torch.Tensor,
    *,
    batch_size: int = 2048,
    amp: bool = False,
) -> torch.Tensor:
    """Logits for uint8 NCHW images already on the model's device."""
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start : start + batch_size])
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
            outputs.append(model(batch).float())
    return torch.cat(outputs)


def fit(
    model: nn.Module,
    preprocess: Preprocessor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: TrainConfig,
    *,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """Train ``model`` in place.

    A fresh shuffle every epoch with the last partial batch kept. With validation data it
    also supports ``ReduceLROnPlateau`` on validation loss and early stopping on validation
    accuracy that restores the best weights.

    Args:
        model: Network already on the target device.
        preprocess: Normalisation applied to every batch.
        X_train: uint8 NCHW training images on the device.
        y_train: Integer labels on the device.
        cfg: Optimisation settings.
        X_val: Optional uint8 validation images (needed for plateau / early stopping).
        y_val: Optional validation labels.

    Returns:
        One dict per epoch with loss, accuracy, validation metrics, learning rate and time.

    Raises:
        ValueError: If a validation-driven option is used without validation data.
    """
    has_val = X_val is not None and y_val is not None
    if (cfg.scheduler == "plateau" or cfg.early_stopping_patience) and not has_val:
        raise ValueError("Plateau scheduling and early stopping need validation data")

    device = X_train.device
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    optimizer = _optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(X_train) / cfg.batch_size)
    scheduler: Any = None
    if cfg.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps_per_epoch
        )
    elif cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience
        )
    elif cfg.scheduler != "none":
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")

    history: list[dict[str, float]] = []
    best_acc, best_state, stale = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.perf_counter()
        order = torch.randperm(len(X_train), device=device, generator=generator)
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(steps_per_epoch):
            idx = order[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            x = preprocess(X_train[idx])
            if cfg.augment:
                x = augment(x, generator)
            target = y_train[idx]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), target, label_smoothing=cfg.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if cfg.scheduler == "onecycle":
                scheduler.step()
            total_loss += loss.detach() * len(idx)
            correct += (logits.argmax(1) == target).sum()

        record = {
            "epoch": epoch,
            "loss": float(total_loss) / len(X_train),
            "accuracy": float(correct) / len(X_train),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if has_val:
            logits = predict_logits(model, preprocess, X_val, amp=cfg.amp)
            record["val_loss"] = float(F.cross_entropy(logits, y_val))
            record["val_accuracy"] = float((logits.argmax(1) == y_val).float().mean())
            if cfg.scheduler == "plateau":
                scheduler.step(record["val_loss"])
        record["seconds"] = time.perf_counter() - start
        history.append(record)
        logger.info(
            "epoch %d/%d %s",
            epoch,
            cfg.epochs,
            " ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"),
        )

        if cfg.early_stopping_patience:
            if record["val_accuracy"] > best_acc:
                best_acc, stale = record["val_accuracy"], 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= cfg.early_stopping_patience:
                    logger.info("early stopping at epoch %d", epoch)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
