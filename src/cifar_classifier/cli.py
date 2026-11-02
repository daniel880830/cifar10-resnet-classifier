"""Command-line interface for downloading data, training, sweeping, serving and benchmarking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from cifar_classifier.config import get_experiment, get_settings, load_params

app = typer.Typer(help="CIFAR-10 image classifier.", no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """CIFAR-10 image classifier."""
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command("download-data")
def download_data(
    keep_archive: Annotated[bool, typer.Option(help="Keep the downloaded .tar.gz.")] = False,
) -> None:
    """Download official CIFAR-10 (binary release), verify its MD5 and write .npy arrays."""
    from cifar_classifier.data import download_cifar10

    settings = get_settings()
    for path in download_cifar10(settings.resolve(settings.data_dir), keep_archive=keep_archive):
        typer.echo(f"wrote {path}")


@app.command("verify-data")
def verify_data() -> None:
    """Check shapes, labels and SHA-256 hashes against data/manifest.json."""
    from cifar_classifier.data import load_arrays, verify_manifest

    settings = get_settings()
    data_dir = settings.resolve(settings.data_dir)
    train, test = load_arrays(data_dir)
    typer.echo(f"train {train.X.shape}  test {test.X.shape}")
    results = verify_manifest(data_dir, settings.resolve(Path("data/manifest.json")))
    for name, ok in results.items():
        typer.echo(f"  {'OK  ' if ok else 'DIFF'} {name}")
    if not all(results.values()):
        raise typer.Exit(code=1)


@app.command()
def train(
    experiment: Annotated[
        list[str], typer.Option(help="Experiment name(s) from params.toml; repeatable.")
    ],
    eval_on: Annotated[str, typer.Option(help="test (final) or val (model selection)")] = "test",
    save: Annotated[bool, typer.Option(help="Save the model artifact.")] = True,
    seeds: Annotated[
        int, typer.Option(min=1, help="Repeat with seeds 0..N-1 and report mean ± std only.")
    ] = 1,
) -> None:
    """Train experiments and write reports/<experiment>[_val].json."""
    from cifar_classifier.data import load_arrays, make_splits
    from cifar_classifier.engine import resolve_device
    from cifar_classifier.evaluate import save_confusion_matrix
    from cifar_classifier.train import run_experiment, save_model, save_report

    if eval_on not in ("test", "val"):
        raise typer.BadParameter("eval-on must be 'test' or 'val'")
    settings = get_settings()
    params = load_params(settings.resolve(settings.params_path))
    experiments = [get_experiment(params, name) for name in experiment]
    device = resolve_device(settings.device)
    splits = make_splits(
        *load_arrays(settings.resolve(settings.data_dir)),
        val_size=params["data"]["val_size"],
        seed=params["data"]["split_seed"],
    )
    reports_dir = settings.resolve(settings.reports_dir)
    if seeds > 1:
        from cifar_classifier.train import run_seeds

        for exp in experiments:
            summary = run_seeds(splits, exp, device, seeds=seeds, eval_on=eval_on)  # type: ignore[arg-type]
            path = reports_dir / f"{exp['name']}_seeds{'_val' if eval_on == 'val' else ''}.json"
            reports_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            acc = summary["accuracy"]
            typer.echo(
                f"{exp['name']:<26} {eval_on} accuracy {acc['mean']:.4f} ± {acc['std']:.4f} "
                f"(min {acc['min']:.4f}, max {acc['max']:.4f}, {seeds} seeds)  report: {path}"
            )
        return
    for exp in experiments:
        result = run_experiment(splits, exp, device, eval_on=eval_on)  # type: ignore[arg-type]
        metrics = result.report["metrics"]
        typer.echo(
            f"{exp['name']:<26} {eval_on} accuracy {metrics['accuracy']:.4f}  "
            f"macro-F1 {metrics['macro_f1']:.4f}  ({result.report['train_seconds']} s, "
            f"{result.report['epochs_run'] or '-'} epochs)"
        )
        typer.echo(f"  report: {save_report(result.report, reports_dir)}")
        if eval_on == "test":
            figure = reports_dir / "figures" / f"confusion_{exp['name']}.png"
            save_confusion_matrix(metrics["confusion_matrix"], f"{exp['name']} (test)", figure)
            if save:
                typer.echo(f"  model:  {save_model(result, settings.resolve(settings.models_dir))}")


@app.command()
def sweep(
    experiment: Annotated[str, typer.Option(help="Base experiment from params.toml.")],
    param: Annotated[
        list[str], typer.Option(help="key=v1,v2,... (repeatable); the grid is their product.")
    ],
) -> None:
    """Grid search on the validation split (the test set is never used)."""
    from cifar_classifier.data import load_arrays, make_splits
    from cifar_classifier.engine import resolve_device
    from cifar_classifier.train import parse_grid, run_sweep

    settings = get_settings()
    params = load_params(settings.resolve(settings.params_path))
    base = get_experiment(params, experiment)
    splits = make_splits(
        *load_arrays(settings.resolve(settings.data_dir)),
        val_size=params["data"]["val_size"],
        seed=params["data"]["split_seed"],
    )
    summary = run_sweep(splits, base, parse_grid(param), resolve_device(settings.device))
    path = settings.resolve(settings.reports_dir) / f"{experiment}_sweep_val.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for run in summary["runs"]:
        typer.echo(
            f"  {run['overrides']}  val accuracy {run['accuracy']:.4f}  ({run['train_seconds']} s)"
        )
    typer.echo(f"best: {summary['best']['overrides']}  report: {path}")


@app.command("export-samples")
def export_samples(
    count: Annotated[int, typer.Option(min=1, max=100)] = 10,
    out_dir: Annotated[Path, typer.Option()] = Path("samples"),
) -> None:
    """Write the first test image of each class as PNG, for trying the API."""
    from PIL import Image

    from cifar_classifier.data import CLASS_NAMES, load_arrays

    settings = get_settings()
    _, test = load_arrays(settings.resolve(settings.data_dir))
    target = settings.resolve(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    for label in range(len(CLASS_NAMES)):
        for index in [int(i) for i in (test.y == label).nonzero()[0][: max(1, count // 10)]]:
            path = target / f"{CLASS_NAMES[label]}_{index}.png"
            Image.fromarray(test.X[index]).save(path)
            written += 1
    typer.echo(f"wrote {written} images to {target}")


@app.command()
def benchmark(
    url: Annotated[str | None, typer.Option(help="Also time a running API at this URL.")] = None,
) -> None:
    """Measure latency/throughput on CPU (and GPU if present); write reports/benchmark.json."""
    from cifar_classifier.benchmark import run_benchmark

    settings = get_settings()
    result = run_benchmark(settings, url=url)
    path = settings.resolve(settings.reports_dir) / "benchmark.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(result, indent=2))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Defaults to CIFAR_API_HOST.")] = None,
    port: Annotated[int | None, typer.Option(help="Defaults to CIFAR_API_PORT.")] = None,
) -> None:
    """Start the inference API."""
    import uvicorn

    from cifar_classifier.api import create_app

    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
