"""Checks against the real dataset (skipped when the .npy files are absent)."""

import numpy as np
import pytest

from cifar_classifier.config import PROJECT_ROOT
from cifar_classifier.data import load_arrays, make_splits, verify_manifest

DATA = PROJECT_ROOT / "data" / "raw"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (DATA / "X_train.npy").is_file(), reason="real dataset not present"),
]


def test_manifest_matches():
    assert all(verify_manifest(DATA, PROJECT_ROOT / "data" / "manifest.json").values())


def test_split_sizes_and_balance():
    splits = make_splits(*load_arrays(DATA), val_size=0.2, seed=42)
    assert (len(splits.train), len(splits.val), len(splits.test)) == (40_000, 10_000, 10_000)
    assert np.bincount(splits.train.y).tolist() == [4000] * 10
    assert np.bincount(splits.val.y).tolist() == [1000] * 10
