"""Classification metrics and report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from cifar_classifier.data import CLASS_NAMES


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro F1, per-class F1, log-loss and the confusion matrix.

    Args:
        y_true: Integer labels ``(N,)``.
        proba: Class probabilities ``(N, 10)``.

    Returns:
        JSON-serialisable metrics.
    """
    y_pred = proba.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1), labels=labels)),
        "per_class_f1": {name: float(v) for name, v in zip(CLASS_NAMES, per_class, strict=True)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_confusion_matrix(matrix: list[list[int]], title: str, path: Path) -> Path:
    """Render a row-normalised confusion matrix to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(matrix, dtype=float)
    norm = counts / counts.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
