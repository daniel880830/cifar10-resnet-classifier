from pathlib import Path

import pytest
from pydantic import ValidationError

from cifar_classifier import config
from cifar_classifier.config import (
    PROJECT_ROOT,
    ConfigError,
    Settings,
    get_experiment,
    load_params,
)


def test_project_params_are_valid():
    params = load_params(PROJECT_ROOT / "config" / "params.toml")
    assert "resnet" in params["experiments"]
    for name in params["experiments"]:
        assert get_experiment(params, name)["model"] == "resnet9"


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_params(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text("[data\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_params(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[data]", "[datax]", "needs"),
        ("val_size = 0.2", "val_size = 1.2", "val_size"),
        ('model = "resnet9"', 'model = "vit"', "model must be"),
        ("epochs = 1", "epochs = 0", "epochs"),
        ('extends = "resnet"', 'extends = "missing"', "Unknown experiment 'missing'"),
    ],
)
def test_invalid_values(params_path: Path, old: str, new: str, message: str):
    params_path.write_text(params_path.read_text().replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_params(params_path)


def test_extends_merges_and_overrides(params: dict):
    base = get_experiment(params, "resnet")
    child = get_experiment(params, "resnet_no_augment")
    assert child["augment"] is False
    assert child["name"] == "resnet_no_augment"
    assert "extends" not in child
    assert {k: v for k, v in child.items() if k not in ("augment", "name")} == {
        k: v for k, v in base.items() if k not in ("augment", "name")
    }


def test_circular_extends(params_path: Path):
    text = params_path.read_text().replace(
        "[experiments.resnet]\n", '[experiments.resnet]\nextends = "resnet_no_augment"\n'
    )
    params_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        load_params(params_path)


def test_get_experiment_unknown(params: dict):
    with pytest.raises(ConfigError, match="Unknown experiment"):
        get_experiment(params, "vit")


def test_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CIFAR_MAX_BATCH_SIZE", "7")
    monkeypatch.setenv("CIFAR_PROJECT_ROOT", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.max_batch_size == 7
    assert settings.resolve(Path("a")) == tmp_path / "a"
    assert settings.resolve(tmp_path) == tmp_path


@pytest.mark.parametrize(
    ("var", "value"),
    [("CIFAR_DEVICE", "tpu"), ("CIFAR_API_PORT", "0"), ("CIFAR_MAX_UPLOAD_BYTES", "10")],
)
def test_settings_reject_invalid(monkeypatch: pytest.MonkeyPatch, var: str, value: str):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_installed_package_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "_SOURCE_ROOT", tmp_path / "site-packages")
    monkeypatch.chdir(tmp_path)
    assert config._default_root() == tmp_path
