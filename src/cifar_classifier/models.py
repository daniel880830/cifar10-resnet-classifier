"""ResNet-9 for 32x32 images."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

NUM_CLASSES = 10


def _conv_bn(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()
    )


class _Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv_bn(channels, channels), _conv_bn(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResNet9(nn.Module):
    """Compact residual network for 32x32 images (the DAWNBench-style ResNet-9)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn(3, 64),
            _conv_bn(64, 128),
            nn.MaxPool2d(2),
            _Residual(128),
            _conv_bn(128, 256),
            nn.MaxPool2d(2),
            _conv_bn(256, 512),
            nn.MaxPool2d(2),
            _Residual(512),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(512, NUM_CLASSES, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits (scaled down for a softer initial softmax)."""
        return self.net(x) * 0.125


def build_torch_model(exp: dict[str, Any]) -> nn.Module:
    """Instantiate the network described by an experiment.

    Raises:
        ValueError: If the model type is unknown.
    """
    if exp["model"] == "resnet9":
        return ResNet9()
    raise ValueError(f"Unknown model '{exp['model']}'")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
