import numpy as np
import pytest
import torch
from torch import nn

from cifar_classifier import engine
from cifar_classifier.engine import (
    Preprocessor,
    TrainConfig,
    augment,
    fit,
    predict_logits,
    resolve_device,
    to_tensor,
)
from cifar_classifier.models import ResNet9, build_torch_model, count_parameters
from tests.conftest import make_arrays


def tiny_net() -> nn.Module:
    """A fast stand-in network for exercising the training loop."""
    return nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 32), nn.ReLU(), nn.Linear(32, 10))


def test_resolve_device(monkeypatch: pytest.MonkeyPatch):
    assert resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_preprocessor_unit_and_standard():
    images = make_arrays(2, 0).X
    x = to_tensor(images, torch.device("cpu"))
    unit = Preprocessor.fit("unit", images)(x)
    assert float(unit.min()) >= 0
    assert float(unit.max()) <= 1
    standard = Preprocessor.fit("standard", images)
    out = standard(x)
    assert out.mean(dim=(0, 2, 3)).abs().max() < 1e-4
    assert torch.allclose(out.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-2)
    assert standard.state()["mode"] == "standard"
    with pytest.raises(ValueError, match="Unknown normalisation"):
        Preprocessor.fit("zca", images)


def test_to_tensor_layout():
    images = make_arrays(1, 0).X
    tensor = to_tensor(images, torch.device("cpu"))
    assert tensor.shape == (10, 3, 32, 32)
    assert tensor.dtype == torch.uint8
    assert int(tensor[3, 1, 5, 7]) == int(images[3, 5, 7, 1])


def test_augment_is_seeded_crop_and_flip():
    x = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).view(2, 3, 32, 32)
    a = augment(x, torch.Generator().manual_seed(1))
    b = augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert torch.equal(a, b)
    assert not torch.equal(a, augment(x, torch.Generator().manual_seed(2)))


def test_resnet_shapes():
    model = build_torch_model({"model": "resnet9"})
    assert isinstance(model, ResNet9)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert count_parameters(model) == 6_573_120


def test_build_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_torch_model({"model": "vit"})


def _tensors(per_class: int = 6):
    data = make_arrays(per_class, 0)
    return to_tensor(data.X, torch.device("cpu")), torch.from_numpy(data.y)


def test_training_config_ignores_unknown_keys():
    cfg = TrainConfig.from_experiment({"epochs": 2, "batch_size": 8, "model": "resnet9"})
    assert (cfg.epochs, cfg.batch_size, cfg.optimizer) == (2, 8, "adam")


@pytest.mark.parametrize(
    "overrides",
    [
        {"optimizer": "adam", "scheduler": "none"},
        {"optimizer": "sgd", "scheduler": "onecycle", "augment": True, "label_smoothing": 0.1},
    ],
)
def test_fit_learns(overrides):
    torch.manual_seed(0)
    X, y = _tensors()
    cfg = TrainConfig(epochs=8, batch_size=16, lr=0.01, **overrides)
    history = fit(tiny_net(), Preprocessor("unit"), X, y, cfg)
    assert len(history) == 8
    assert history[-1]["loss"] < history[0]["loss"]


def test_fit_plateau_and_early_stopping_restore_best(monkeypatch: pytest.MonkeyPatch):
    X, y = _tensors()
    accuracies = iter([0.5, 0.4, 0.3, 0.2, 0.1])
    real = engine.predict_logits

    def fake_predict(m, p, images, **kw):
        logits = real(m, p, images, **kw)
        correct = int(next(accuracies) * len(images))
        out = torch.full_like(logits, -10.0)
        out[torch.arange(correct), y[:correct]] = 10.0
        return out

    monkeypatch.setattr(engine, "predict_logits", fake_predict)
    cfg = TrainConfig(
        epochs=5, batch_size=16, scheduler="plateau", plateau_patience=0, early_stopping_patience=2
    )
    history = fit(tiny_net(), Preprocessor("unit"), X, y, cfg, X_val=X, y_val=y)
    assert len(history) == 3  # best at epoch 1, two stale epochs
    assert history[-1]["lr"] < cfg.lr


def test_fit_validation_requirements():
    X, y = _tensors(1)
    cfg = TrainConfig(epochs=1, batch_size=4, scheduler="plateau")
    with pytest.raises(ValueError, match="need validation data"):
        fit(tiny_net(), Preprocessor(), X, y, cfg)


@pytest.mark.parametrize(("key", "value"), [("optimizer", "rmsprop"), ("scheduler", "cosine")])
def test_fit_rejects_unknown_options(key, value):
    X, y = _tensors(1)
    cfg = TrainConfig(epochs=1, batch_size=4, **{key: value})
    with pytest.raises(ValueError, match="Unknown"):
        fit(tiny_net(), Preprocessor(), X, y, cfg)


def test_predict_logits_batches_consistently():
    X, _ = _tensors(2)
    model = ResNet9().eval()
    full = predict_logits(model, Preprocessor(), X, batch_size=1000)
    small = predict_logits(model, Preprocessor(), X, batch_size=3)
    np.testing.assert_allclose(full.numpy(), small.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resnet_trains_on_gpu_with_bf16():
    device = torch.device("cuda")
    X, y = _tensors()
    model = ResNet9().to(device)
    cfg = TrainConfig(epochs=1, batch_size=32, optimizer="sgd", lr=0.05, augment=True, amp=True)
    history = fit(model, Preprocessor("unit"), X.to(device), y.to(device), cfg)
    assert np.isfinite(history[0]["loss"])
