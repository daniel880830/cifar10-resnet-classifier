import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cifar_classifier import data
from cifar_classifier.data import (
    FILES,
    DataValidationError,
    download_cifar10,
    extract_arrays,
    file_digest,
    load_arrays,
    make_splits,
    parse_binary_batch,
    validate,
    verify_manifest,
)
from tests.conftest import make_archive, make_arrays, to_binary


def test_load_arrays(data_dir: Path):
    train, test = load_arrays(data_dir)
    assert train.X.shape == (100, 32, 32, 3)
    assert len(test) == 40
    assert train.y.dtype == np.int64


def test_missing_file(data_dir: Path):
    (data_dir / "y_test.npy").unlink()
    with pytest.raises(FileNotFoundError, match="download-data"):
        load_arrays(data_dir)


def test_unreadable_array(data_dir: Path):
    (data_dir / "X_test.npy").write_bytes(b"garbage")
    with pytest.raises(DataValidationError, match="Could not read"):
        load_arrays(data_dir)


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.zeros((2, 32, 32, 3), np.float32), np.zeros(2, int), "uint8"),
        (np.zeros((2, 28, 28, 3), np.uint8), np.zeros(2, int), "expected"),
        (np.zeros((2, 32, 32, 3), np.uint8), np.zeros(3, int), "labels"),
        (np.zeros((0, 32, 32, 3), np.uint8), np.zeros(0, int), "no samples"),
        (np.zeros((2, 32, 32, 3), np.uint8), np.array([0, 10]), r"\[0, 9\]"),
        (np.zeros((2, 32, 32, 3), np.uint8), np.array([0.0, 1.0]), r"\[0, 9\]"),
    ],
)
def test_validate_rejects(X, y, message):
    with pytest.raises(DataValidationError, match=message):
        validate(X, y, "x")


def test_splits_are_stratified_and_disjoint(data_dir: Path):
    splits = make_splits(*load_arrays(data_dir), val_size=0.2, seed=42)
    assert (len(splits.train), len(splits.val)) == (80, 20)
    assert np.bincount(splits.val.y).tolist() == [2] * 10
    fingerprint = lambda a: {tuple(x[0, 0]) for x in a.X}  # noqa: E731
    assert not fingerprint(splits.train) & fingerprint(splits.val)
    assert len(splits.train_val) == 100


def test_splits_are_reproducible(data_dir: Path):
    a = make_splits(*load_arrays(data_dir), val_size=0.2, seed=42)
    b = make_splits(*load_arrays(data_dir), val_size=0.2, seed=42)
    np.testing.assert_array_equal(a.val.X, b.val.X)


def test_manifest(data_dir: Path, tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({f: file_digest(data_dir / f) for f in FILES}))
    assert all(verify_manifest(data_dir, manifest).values())
    (data_dir / "y_test.npy").unlink()
    assert verify_manifest(data_dir, manifest)["y_test.npy"] is False


def test_parse_binary_round_trip():
    arrays = make_arrays(2, 0)
    parsed = parse_binary_batch(to_binary(arrays), "batch")
    np.testing.assert_array_equal(parsed.X, arrays.X)
    np.testing.assert_array_equal(parsed.y, arrays.y)


@pytest.mark.parametrize("raw", [b"", b"x" * 100])
def test_parse_binary_rejects_partial_records(raw: bytes):
    with pytest.raises(DataValidationError, match="multiple"):
        parse_binary_batch(raw, "batch")


def test_extract_arrays(tmp_path: Path):
    train, test = make_arrays(5, 0), make_arrays(2, 1)
    got_train, got_test = extract_arrays(make_archive(tmp_path / "c.tar.gz", train, test))
    np.testing.assert_array_equal(got_train.y, train.y)
    np.testing.assert_array_equal(got_test.X, test.X)


def test_extract_arrays_missing_batch(tmp_path: Path):
    archive = make_archive(
        tmp_path / "c.tar.gz", make_arrays(5, 0), make_arrays(2, 1), drop="data_batch_3.bin"
    )
    with pytest.raises(DataValidationError, match=r"data_batch_3.bin"):
        extract_arrays(archive)


def _fake_download(monkeypatch: pytest.MonkeyPatch, source: Path) -> list[str]:
    calls: list[str] = []

    def urlretrieve(url: str, dest: Path) -> None:
        calls.append(url)
        Path(dest).write_bytes(source.read_bytes())

    monkeypatch.setattr(data.urllib.request, "urlretrieve", urlretrieve)
    return calls


def test_download_verifies_md5_and_writes_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = make_archive(tmp_path / "src.tar.gz", make_arrays(5, 0), make_arrays(2, 1))
    monkeypatch.setattr(data, "CIFAR10_MD5", hashlib.md5(source.read_bytes()).hexdigest())
    calls = _fake_download(monkeypatch, source)
    written = download_cifar10(tmp_path / "out")
    assert calls == [data.CIFAR10_URL]
    assert sorted(p.name for p in written) == sorted(FILES)
    assert load_arrays(tmp_path / "out")[0].X.shape == (50, 32, 32, 3)
    assert not (tmp_path / "out" / "cifar-10-binary.tar.gz").exists()


def test_download_rejects_bad_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = make_archive(tmp_path / "src.tar.gz", make_arrays(5, 0), make_arrays(2, 1))
    _fake_download(monkeypatch, source)
    with pytest.raises(DataValidationError, match="MD5 mismatch"):
        download_cifar10(tmp_path / "out", md5="0" * 32)
    assert not (tmp_path / "out" / "cifar-10-binary.tar.gz").exists()


def test_download_reuses_existing_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    out.mkdir()
    archive = make_archive(out / "cifar-10-binary.tar.gz", make_arrays(5, 0), make_arrays(2, 1))
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    calls = _fake_download(monkeypatch, archive)
    download_cifar10(out, md5=md5, keep_archive=True)
    assert calls == []
    assert archive.exists()
