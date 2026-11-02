import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cifar_classifier.config import get_experiment
from cifar_classifier.predict import (
    ImageDecodeError,
    ModelLoadError,
    ModelNotFoundError,
    decode_image,
    load_model,
    predict_images,
)
from cifar_classifier.train import (
    parse_grid,
    run_experiment,
    run_seeds,
    run_sweep,
    save_model,
    save_report,
)
from tests.conftest import png_bytes


def test_experiment_trains_on_all_training_images(splits, params, cpu, tmp_path: Path):
    result = run_experiment(splits, get_experiment(params, "resnet"), cpu)
    report = result.report
    assert report["train_samples"] == len(splits.train) + len(splits.val) == 100
    assert report["eval_samples"] == len(splits.test)
    assert report["epochs_run"] == 1
    assert len(report["metrics"]["confusion_matrix"]) == 10
    assert save_model(result, tmp_path).suffix == ".pt"


def test_eval_on_val_never_uses_test(splits, params, cpu):
    report = run_experiment(splits, get_experiment(params, "resnet"), cpu, eval_on="val").report
    assert report["evaluated_on"] == "val"
    assert report["eval_samples"] == len(splits.val)
    assert report["train_samples"] == len(splits.train)


def test_extended_experiment_config_is_recorded(splits, params, cpu):
    report = run_experiment(splits, get_experiment(params, "resnet_no_augment"), cpu).report
    assert report["config"]["augment"] is False
    assert "extends" not in report["config"]


def test_cpu_runs_are_reproducible(splits, params, cpu):
    exp = get_experiment(params, "resnet")
    a = run_experiment(splits, exp, cpu).report["metrics"]
    b = run_experiment(splits, exp, cpu).report["metrics"]
    assert a == b


def test_save_report_names(tmp_path: Path, splits, params, cpu):
    report = run_experiment(splits, get_experiment(params, "resnet"), cpu, eval_on="val").report
    assert save_report(report, tmp_path).name == "resnet_val.json"
    report["evaluated_on"] = "test"
    assert json.loads(save_report(report, tmp_path).read_text())["experiment"] == "resnet"


def test_run_seeds(splits, params, cpu):
    summary = run_seeds(splits, get_experiment(params, "resnet"), cpu, seeds=2)
    assert [r["seed"] for r in summary["runs"]] == [0, 1]
    assert summary["accuracy"]["min"] <= summary["accuracy"]["mean"] <= summary["accuracy"]["max"]


def test_parse_grid():
    assert parse_grid(["lr=0.1,0.2", "epochs=3", "optimizer=sgd,adam"]) == {
        "lr": [0.1, 0.2],
        "epochs": [3],
        "optimizer": ["sgd", "adam"],
    }
    with pytest.raises(ValueError, match="key=v1"):
        parse_grid(["lr"])


def test_run_sweep(splits, params, cpu):
    summary = run_sweep(splits, get_experiment(params, "resnet"), {"lr": [0.01, 0.05]}, cpu)
    assert len(summary["runs"]) == 2
    assert summary["best"] in summary["runs"]
    assert summary["evaluated_on"] == "val"


# ---------------------------------------------------------------- prediction


def test_load_and_predict(model_path: Path, cpu, splits):
    model = load_model(model_path, cpu)
    predictions = predict_images(model, splits.test.X[:5])
    assert len(predictions) == 5
    first = predictions[0]
    assert first.class_name == model.class_names[first.label]
    probabilities = [s.probability for s in first.top_k]
    assert probabilities == sorted(probabilities, reverse=True)
    assert first.top_k[0].probability == first.probability


def test_single_matches_batch(model_path: Path, cpu, splits):
    model = load_model(model_path, cpu)
    batch = predict_images(model, splits.test.X[:4])
    singles = [predict_images(model, splits.test.X[i : i + 1])[0] for i in range(4)]
    assert [p.label for p in batch] == [p.label for p in singles]
    np.testing.assert_allclose(
        [p.probability for p in batch], [p.probability for p in singles], rtol=1e-4
    )


@pytest.mark.parametrize("shape", [(0, 32, 32, 3), (2, 32, 32), (1, 28, 28, 3)])
def test_predict_rejects_bad_shapes(model_path: Path, cpu, shape):
    with pytest.raises(ValueError, match="Expected"):
        predict_images(load_model(model_path, cpu), np.zeros(shape, np.uint8))


def test_load_model_errors(tmp_path: Path, cpu):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "none.pt", cpu)
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"garbage")
    with pytest.raises(ModelLoadError):
        load_model(corrupt, cpu)
    wrong = tmp_path / "wrong.pt"
    torch.save({"state_dict": {}, "metadata": {"model": "resnet9", "config": {}}}, wrong)
    with pytest.raises(ModelLoadError):
        load_model(wrong, cpu)


def test_load_model_refuses_arbitrary_pickles(tmp_path: Path, cpu):
    path = tmp_path / "evil.pt"
    torch.save({"state_dict": {}, "metadata": object()}, path)
    with pytest.raises(ModelLoadError):
        load_model(path, cpu)


@pytest.mark.parametrize(
    "data",
    [png_bytes(), png_bytes((64, 48)), png_bytes(mode="L"), png_bytes(fmt="JPEG")],
)
def test_decode_image(data: bytes):
    image = decode_image(data)
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("data", [b"", b"not an image", png_bytes()[:20]])
def test_decode_image_rejects(data: bytes):
    with pytest.raises(ImageDecodeError):
        decode_image(data)
