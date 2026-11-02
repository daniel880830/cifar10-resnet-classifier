from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cifar_classifier.config import get_settings, load_params
from cifar_classifier.data import RECORD_BYTES, Arrays, Splits, load_arrays, make_splits
from cifar_classifier.train import run_experiment, save_model

TINY_PARAMS = """
[data]
val_size = 0.2
split_seed = 42

[experiments.resnet]
model = "resnet9"
normalize = "standard"
augment = true
optimizer = "sgd"
nesterov = true
momentum = 0.9
lr = 0.05
weight_decay = 5e-4
batch_size = 64
epochs = 1
scheduler = "onecycle"
label_smoothing = 0.1
seed = 0

[experiments.resnet_no_augment]
extends = "resnet"
augment = false
"""


def make_arrays(per_class: int, seed: int) -> Arrays:
    """Class k images are brighter in channel k % 3, so a model can learn something."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(10), per_class)
    X = rng.integers(0, 120, size=(len(y), 32, 32, 3), dtype=np.uint8)
    for i, label in enumerate(y):
        X[i, :, :, label % 3] += np.uint8(60 + 10 * (label // 3))
        X[i, 0, 0, :] = [i % 256, i // 256, 0]  # unique fingerprint per image
    return Arrays(X, y.astype(np.int64))


def write_arrays(data_dir: Path, train: Arrays, test: Arrays) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "X_train.npy", train.X)
    np.save(data_dir / "y_train.npy", train.y)
    np.save(data_dir / "X_test.npy", test.X)
    np.save(data_dir / "y_test.npy", test.y)
    return data_dir


def to_binary(arrays: Arrays) -> bytes:
    """Encode arrays in the official CIFAR-10 binary record format."""
    planes = arrays.X.transpose(0, 3, 1, 2).reshape(len(arrays), -1)
    records = np.concatenate([arrays.y.astype(np.uint8)[:, None], planes], axis=1)
    assert records.shape[1] == RECORD_BYTES
    return records.tobytes()


def make_archive(path: Path, train: Arrays, test: Arrays, *, drop: str | None = None) -> Path:
    """A fake ``cifar-10-binary.tar.gz`` with five train batches and one test batch."""
    chunks = np.array_split(np.arange(len(train)), 5)
    members = {
        f"data_batch_{i + 1}.bin": Arrays(train.X[c], train.y[c]) for i, c in enumerate(chunks)
    }
    members["test_batch.bin"] = test
    with tarfile.open(path, "w:gz") as tar:
        for name, arrays in members.items():
            if name == drop:
                continue
            data = to_binary(arrays)
            info = tarfile.TarInfo(f"cifar-10-batches-bin/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return write_arrays(tmp_path / "data", make_arrays(10, 0), make_arrays(4, 1))


@pytest.fixture
def params_path(tmp_path: Path) -> Path:
    path = tmp_path / "params.toml"
    path.write_text(TINY_PARAMS, encoding="utf-8")
    return path


@pytest.fixture
def params(params_path: Path) -> dict:
    return load_params(params_path)


@pytest.fixture
def splits(data_dir: Path) -> Splits:
    return make_splits(*load_arrays(data_dir), val_size=0.2, seed=42)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def model_path(tmp_path: Path, splits: Splits, params: dict, cpu: torch.device) -> Path:
    exp = {"name": "resnet", **params["experiments"]["resnet"]}
    return save_model(run_experiment(splits, exp, cpu), tmp_path / "models")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_dir: Path, params_path: Path):
    """Point every CIFAR_* path at temporary files and reset the settings cache."""
    monkeypatch.setenv("CIFAR_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CIFAR_PARAMS_PATH", str(params_path))
    monkeypatch.setenv("CIFAR_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CIFAR_MODEL_PATH", str(tmp_path / "models" / "resnet.pt"))
    monkeypatch.setenv("CIFAR_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("CIFAR_DEVICE", "cpu")
    monkeypatch.setenv("CIFAR_LOG_LEVEL", "warning")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def png_bytes(size: tuple[int, int] = (32, 32), mode: str = "RGB", fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=(200, 30, 30) if mode == "RGB" else 128).save(buffer, format=fmt)
    return buffer.getvalue()
