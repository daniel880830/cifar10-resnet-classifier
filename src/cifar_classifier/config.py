"""Runtime settings (environment / .env) and experiment parameters (TOML)."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SOURCE_ROOT = Path(__file__).resolve().parents[2]

MODEL_TYPES = ("resnet9",)


def _default_root() -> Path:
    """The source checkout when running from it, otherwise the working directory."""
    return _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()


PROJECT_ROOT = _default_root()


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


class Settings(BaseSettings):
    """Runtime settings read from environment variables prefixed with ``CIFAR_``."""

    model_config = SettingsConfigDict(env_prefix="CIFAR_", env_file=".env", extra="ignore")

    project_root: Path = PROJECT_ROOT
    data_dir: Path = Path("data/raw")
    params_path: Path = Path("config/params.toml")
    models_dir: Path = Path("models")
    model_path: Path = Path("models/resnet.pt")
    reports_dir: Path = Path("reports")
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_batch_size: int = Field(default=64, ge=1)
    max_upload_bytes: int = Field(default=2_000_000, ge=1_000)
    log_level: str = Field(default="info", pattern="^(critical|error|warning|info|debug)$")

    def resolve(self, path: Path) -> Path:
        """Return ``path`` as an absolute path anchored at ``project_root``."""
        return path if path.is_absolute() else self.project_root / path


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()


def load_params(path: Path) -> dict[str, Any]:
    """Load and validate the experiment file.

    Args:
        path: Path to ``params.toml``.

    Returns:
        The parsed parameters.

    Raises:
        ConfigError: If the file is missing, unparsable or an experiment is invalid.
    """
    if not path.is_file():
        raise ConfigError(f"Parameter file not found: {path}")
    try:
        params = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    if "data" not in params or "experiments" not in params:
        raise ConfigError(f"{path} needs [data] and [experiments.*] sections")
    if not 0 < params["data"].get("val_size", 0) < 1:
        raise ConfigError("data.val_size must be between 0 and 1")
    for name in params["experiments"]:
        exp = get_experiment(params, name)
        if exp.get("model") not in MODEL_TYPES:
            raise ConfigError(f"experiments.{name}.model must be one of {MODEL_TYPES}")
        if exp.get("epochs", 0) < 1:
            raise ConfigError(f"experiments.{name}.epochs must be >= 1")
    return params


def get_experiment(params: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one experiment's settings, resolving ``extends``, with its name attached.

    Raises:
        ConfigError: If the experiment (or one it extends) is undefined or the chain loops.
    """
    experiments = params["experiments"]
    chain: list[str] = []
    merged: dict[str, Any] = {}
    current: str | None = name
    while current is not None:
        if current not in experiments:
            raise ConfigError(f"Unknown experiment '{current}'. Choose from {sorted(experiments)}")
        if current in chain:
            raise ConfigError(f"Circular 'extends' in experiments: {[*chain, current]}")
        chain.append(current)
        merged = {**experiments[current], **merged}
        current = experiments[current].get("extends")
    merged.pop("extends", None)
    return {"name": name, **merged}
