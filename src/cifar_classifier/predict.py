"""Loading a trained network and classifying images."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from cifar_classifier.engine import Preprocessor, predict_logits, to_tensor
from cifar_classifier.models import build_torch_model
from cifar_classifier.schema import ClassScore, Prediction

TOP_K = 3


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model artifact does not exist."""


class ModelLoadError(RuntimeError):
    """Raised when the artifact exists but cannot be used."""


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a readable image."""


@dataclass
class InferenceModel:
    """A network in eval mode with its preprocessing and metadata."""

    network: nn.Module
    preprocessor: Preprocessor
    metadata: dict[str, Any]
    device: torch.device

    @property
    def class_names(self) -> list[str]:
        """Class names indexed by label."""
        return list(self.metadata["class_names"])


def load_model(path: Path, device: torch.device) -> InferenceModel:
    """Load a ``.pt`` artifact written by ``train.save_model``.

    ``weights_only=True`` means only tensors and plain containers are unpickled, so a
    tampered file cannot execute code on load.

    Raises:
        ModelNotFoundError: If ``path`` does not exist.
        ModelLoadError: If the file is corrupt or not a compatible artifact.
    """
    if not path.is_file():
        raise ModelNotFoundError(f"Model artifact not found: {path}. Run `cifar train` first.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        network = build_torch_model({**metadata["config"], "model": metadata["model"]})
        network.load_state_dict(payload["state_dict"])
        preprocessor = Preprocessor(**metadata["preprocessor"])
    except Exception as exc:  # pickle / key / shape errors all mean "unusable artifact"
        raise ModelLoadError(f"Could not load model artifact {path}: {exc}") from exc
    network.to(device).eval()
    return InferenceModel(network, preprocessor, metadata, device)


def decode_image(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG/... bytes into a uint8 ``(32, 32, 3)`` RGB array.

    Larger images are resized to 32x32 (the model's native resolution).

    Raises:
        ImageDecodeError: If the bytes are empty or not an image.
    """
    if not data:
        raise ImageDecodeError("Empty file")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            if image.size != (32, 32):
                image = image.resize((32, 32), Image.Resampling.BICUBIC)
            return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"Not a readable image: {exc}") from exc


def predict_images(model: InferenceModel, images: np.ndarray) -> list[Prediction]:
    """Classify uint8 ``(N, 32, 32, 3)`` images.

    Raises:
        ValueError: If ``images`` is empty or has the wrong shape.
    """
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3) or len(images) == 0:
        raise ValueError(f"Expected a non-empty (N, 32, 32, 3) array, got {images.shape}")
    logits = predict_logits(model.network, model.preprocessor, to_tensor(images, model.device))
    proba = torch.softmax(logits, 1).cpu().numpy()
    names = model.class_names
    predictions = []
    for row in proba:
        order = np.argsort(row)[::-1][:TOP_K]
        predictions.append(
            Prediction(
                label=int(order[0]),
                class_name=names[order[0]],
                probability=float(row[order[0]]),
                top_k=[ClassScore(class_name=names[i], probability=float(row[i])) for i in order],
            )
        )
    return predictions
