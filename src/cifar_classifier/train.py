"""Running an experiment end to end and persisting its artifacts."""

from __future__ import annotations

import itertools
import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from cifar_classifier.data import CLASS_NAMES, Splits
from cifar_classifier.engine import (
    Preprocessor,
    TrainConfig,
    fit,
    predict_logits,
    seed_everything,
    to_tensor,
)
from cifar_classifier.evaluate import classification_metrics
from cifar_classifier.models import build_torch_model, count_parameters

logger = logging.getLogger(__name__)

EvalOn = Literal["test", "val"]


@dataclass
class ExperimentResult:
    """Metrics, training history and the fitted model of one run."""

    report: dict[str, Any]
    model: nn.Module
    preprocessor: Preprocessor


def run_experiment(
    splits: Splits, exp: dict[str, Any], device: torch.device, *, eval_on: EvalOn = "test"
) -> ExperimentResult:
    """Train one experiment and evaluate it.

    ``eval_on="val"`` is for model selection: train on the 40k split and score on the
    validation split, never touching the test set. ``eval_on="test"`` trains on all 50k
    training images and scores the official test set.

    Args:
        splits: Train / validation / test data.
        exp: Experiment settings from ``params.toml`` (with ``name``).
        device: Torch device.
        eval_on: ``test`` for final numbers, ``val`` for model selection.

    Returns:
        The report and the fitted model.
    """
    seed_everything(exp.get("seed", 0))
    train, evaluation = (
        (splits.train, splits.val) if eval_on == "val" else (splits.train_val, splits.test)
    )
    start = time.perf_counter()

    preprocessor = Preprocessor.fit(exp.get("normalize", "unit"), train.X)
    model = build_torch_model(exp).to(device).to(memory_format=torch.channels_last)
    cfg = TrainConfig.from_experiment(exp)
    history = fit(
        model, preprocessor, to_tensor(train.X, device), torch.from_numpy(train.y).to(device), cfg
    )
    logits = predict_logits(model, preprocessor, to_tensor(evaluation.X, device), amp=cfg.amp)
    proba = torch.softmax(logits, 1).cpu().numpy()

    report = {
        "experiment": exp["name"],
        "model": exp["model"],
        "evaluated_on": eval_on,
        "train_samples": len(train),
        "eval_samples": len(evaluation),
        "metrics": classification_metrics(evaluation.y, proba),
        "train_seconds": round(time.perf_counter() - start, 1),
        "epochs_run": len(history),
        "parameters": count_parameters(model),
        "config": {k: v for k, v in exp.items() if k != "name"},
        "history": history,
        "environment": environment(device),
    }
    logger.info("%s on %s: accuracy %.4f", exp["name"], eval_on, report["metrics"]["accuracy"])
    return ExperimentResult(report=report, model=model, preprocessor=preprocessor)


def run_seeds(
    splits: Splits,
    exp: dict[str, Any],
    device: torch.device,
    *,
    seeds: int,
    eval_on: EvalOn = "test",
) -> dict[str, Any]:
    """Repeat an experiment with seeds ``0..seeds-1`` to measure run-to-run variance.

    Returns:
        Per-seed accuracy / macro-F1 and their mean, std, min and max.
    """
    runs = []
    for seed in range(seeds):
        report = run_experiment(splits, {**exp, "seed": seed}, device, eval_on=eval_on).report
        runs.append(
            {
                "seed": seed,
                "accuracy": report["metrics"]["accuracy"],
                "macro_f1": report["metrics"]["macro_f1"],
                "train_seconds": report["train_seconds"],
            }
        )

    def stats(key: str) -> dict[str, float]:
        values = np.array([r[key] for r in runs])
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "experiment": exp["name"],
        "evaluated_on": eval_on,
        "seeds": seeds,
        "accuracy": stats("accuracy"),
        "macro_f1": stats("macro_f1"),
        "runs": runs,
        "environment": environment(device),
    }


def parse_grid(specs: list[str]) -> dict[str, list[Any]]:
    """Parse ``["lr=0.1,0.2", "epochs=30"]`` into ``{"lr": [0.1, 0.2], "epochs": [30]}``.

    Raises:
        ValueError: If a spec is not ``key=value[,value...]``.
    """
    grid: dict[str, list[Any]] = {}
    for spec in specs:
        key, sep, values = spec.partition("=")
        if not sep or not key or not values:
            raise ValueError(f"Expected key=v1,v2,... but got '{spec}'")
        parsed: list[Any] = []
        for raw in values.split(","):
            try:
                number = float(raw)
                parsed.append(int(number) if number.is_integer() and "." not in raw else number)
            except ValueError:
                parsed.append(raw)
        grid[key.strip()] = parsed
    return grid


def run_sweep(
    splits: Splits, base: dict[str, Any], grid: dict[str, list[Any]], device: torch.device
) -> dict[str, Any]:
    """Evaluate every combination of ``grid`` on the validation split.

    Returns:
        All runs and the best one by validation accuracy.
    """
    keys = list(grid)
    runs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, values, strict=True))
        report = run_experiment(splits, {**base, **overrides}, device, eval_on="val").report
        runs.append(
            {
                "overrides": overrides,
                "accuracy": report["metrics"]["accuracy"],
                "macro_f1": report["metrics"]["macro_f1"],
                "train_seconds": report["train_seconds"],
            }
        )
    return {
        "experiment": base["name"],
        "evaluated_on": "val",
        "grid": grid,
        "runs": runs,
        "best": max(runs, key=lambda r: r["accuracy"]),
        "environment": environment(device),
    }


def environment(device: torch.device) -> dict[str, str]:
    """Library versions and hardware used for a run."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
    }


def save_report(report: dict[str, Any], reports_dir: Path) -> Path:
    """Write ``<experiment>[_val].json``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_val" if report["evaluated_on"] == "val" else ""
    path = reports_dir / f"{report['experiment']}{suffix}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def save_model(result: ExperimentResult, models_dir: Path) -> Path:
    """Persist ``<experiment>.pt``: CPU state dict plus preprocessing and metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    report = result.report
    metadata = {
        "experiment": report["experiment"],
        "model": report["model"],
        "config": report["config"],
        "class_names": list(CLASS_NAMES),
        "test_metrics": {k: report["metrics"][k] for k in ("accuracy", "macro_f1", "log_loss")},
        "evaluated_on": report["evaluated_on"],
        "environment": report["environment"],
        "preprocessor": result.preprocessor.state(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = models_dir / f"{report['experiment']}.pt"
    state = {k: v.detach().cpu() for k, v in result.model.state_dict().items()}
    torch.save({"state_dict": state, "metadata": metadata}, path)
    return path
