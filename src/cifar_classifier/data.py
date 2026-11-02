"""Downloading, validating and splitting CIFAR-10."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
FILES = ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")
IMAGE_SHAPE = (32, 32, 3)

# Official binary release (no pickle involved) and its published MD5 checksum.
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
TRAIN_BATCHES = tuple(f"data_batch_{i}.bin" for i in range(1, 6))
TEST_BATCH = "test_batch.bin"
RECORD_BYTES = 1 + 32 * 32 * 3


class DataValidationError(ValueError):
    """Raised when the arrays or the downloaded archive are not as expected."""


@dataclass(frozen=True)
class Arrays:
    """Images as uint8 ``(N, 32, 32, 3)`` and integer labels ``(N,)``."""

    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class Splits:
    """Train / validation / test arrays. ``train`` + ``val`` = the official training set."""

    train: Arrays
    val: Arrays
    test: Arrays

    @property
    def train_val(self) -> Arrays:
        """Train and validation concatenated, i.e. all 50k training images."""
        return Arrays(
            np.concatenate([self.train.X, self.val.X]), np.concatenate([self.train.y, self.val.y])
        )


def validate(X: np.ndarray, y: np.ndarray, name: str) -> Arrays:
    """Check dtype, shape and label range.

    Raises:
        DataValidationError: On any mismatch.
    """
    if X.dtype != np.uint8:
        raise DataValidationError(f"{name}: images must be uint8, got {X.dtype}")
    if X.ndim != 4 or X.shape[1:] != IMAGE_SHAPE:
        raise DataValidationError(f"{name}: expected (N, 32, 32, 3) images, got {X.shape}")
    if y.ndim != 1 or len(y) != len(X):
        raise DataValidationError(f"{name}: {len(X)} images but labels have shape {y.shape}")
    if len(y) == 0:
        raise DataValidationError(f"{name}: no samples")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= len(CLASS_NAMES):
        raise DataValidationError(f"{name}: labels must be integers in [0, 9]")
    return Arrays(X, y.astype(np.int64))


def load_arrays(data_dir: Path) -> tuple[Arrays, Arrays]:
    """Load the train and test arrays written by ``download_cifar10``.

    Raises:
        FileNotFoundError: If a file is missing (run ``cifar download-data``).
        DataValidationError: If an array is malformed.
    """
    missing = [f for f in FILES if not (data_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {data_dir}. Run `cifar download-data`.")
    try:
        loaded = {f: np.load(data_dir / f, allow_pickle=False) for f in FILES}
    except ValueError as exc:
        raise DataValidationError(f"Could not read arrays in {data_dir}: {exc}") from exc
    train = validate(loaded["X_train.npy"], loaded["y_train.npy"], "train")
    test = validate(loaded["X_test.npy"], loaded["y_test.npy"], "test")
    return train, test


def make_splits(train: Arrays, test: Arrays, *, val_size: float, seed: int) -> Splits:
    """Stratified train/validation split of the official training set."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        train.X, train.y, test_size=val_size, stratify=train.y, random_state=seed
    )
    return Splits(Arrays(X_tr, y_tr), Arrays(X_val, y_val), test)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Streaming hash of a file."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(data_dir: Path, manifest: Path) -> dict[str, bool]:
    """Compare file hashes with a committed manifest; returns ``{file: matches}``."""
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        name: (data_dir / name).is_file() and file_digest(data_dir / name) == digest
        for name, digest in expected.items()
    }


def parse_binary_batch(raw: bytes, name: str) -> Arrays:
    """Decode one CIFAR-10 binary batch: per record, 1 label byte + 3072 bytes (R, G, B planes).

    Raises:
        DataValidationError: If the byte length is not a whole number of records.
    """
    if not raw or len(raw) % RECORD_BYTES:
        raise DataValidationError(f"{name}: {len(raw)} bytes is not a multiple of {RECORD_BYTES}")
    records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, RECORD_BYTES)
    images = records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return validate(np.ascontiguousarray(images), records[:, 0].astype(np.int64), name)


def extract_arrays(archive: Path) -> tuple[Arrays, Arrays]:
    """Read the train and test batches out of the binary ``.tar.gz`` without extracting to disk.

    Raises:
        DataValidationError: If a batch is missing from the archive.
    """
    batches: dict[str, Arrays] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in (*TRAIN_BATCHES, TEST_BATCH):
                handle = tar.extractfile(member)
                assert handle is not None  # regular files always have a stream
                batches[name] = parse_binary_batch(handle.read(), name)
    missing = [n for n in (*TRAIN_BATCHES, TEST_BATCH) if n not in batches]
    if missing:
        raise DataValidationError(f"{archive.name} is missing {missing}")
    train = Arrays(
        np.concatenate([batches[n].X for n in TRAIN_BATCHES]),
        np.concatenate([batches[n].y for n in TRAIN_BATCHES]),
    )
    return train, batches[TEST_BATCH]


def download_cifar10(
    data_dir: Path,
    *,
    url: str | None = None,
    md5: str | None = None,
    keep_archive: bool = False,
) -> list[Path]:
    """Download the official binary release, verify its MD5 and write the four ``.npy`` files.

    Args:
        data_dir: Destination directory.
        url: Archive URL; defaults to ``CIFAR10_URL`` (overridable for mirrors and tests).
        md5: Expected MD5; defaults to ``CIFAR10_MD5``.
        keep_archive: Keep the downloaded ``.tar.gz`` next to the arrays.

    Returns:
        Paths of the written ``.npy`` files.

    Raises:
        DataValidationError: If the checksum does not match or the archive is incomplete.
    """
    url, md5 = url or CIFAR10_URL, md5 or CIFAR10_MD5
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / Path(url).name
    if not archive.is_file():
        urllib.request.urlretrieve(url, archive)
    actual = file_digest(archive, "md5")
    if actual != md5:
        archive.unlink()
        raise DataValidationError(f"MD5 mismatch for {archive.name}: expected {md5}, got {actual}")
    train, test = extract_arrays(archive)
    arrays = {"X_train": train.X, "y_train": train.y, "X_test": test.X, "y_test": test.y}
    written = []
    for name, array in arrays.items():
        np.save(data_dir / f"{name}.npy", array)
        written.append(data_dir / f"{name}.npy")
    if not keep_archive:
        archive.unlink()
    return written
