"""Latency and throughput of the served model, in-process and over HTTP."""

from __future__ import annotations

import io
import platform
import time
import urllib.request
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image

from cifar_classifier.config import Settings
from cifar_classifier.data import load_arrays
from cifar_classifier.predict import load_model, predict_images


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms)
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


def time_device(
    settings: Settings, device: torch.device, images: np.ndarray, *, n_single: int = 200
) -> dict[str, Any]:
    """Single-image latency and best-of-5 throughput for one batch of ``images``."""
    model = load_model(settings.resolve(settings.model_path), device)
    predict_images(model, images[:8])  # warm-up (CUDA context, cuDNN autotune)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    single = []
    for i in range(n_single):
        index = i % len(images)
        start = time.perf_counter()
        predict_images(model, images[index : index + 1])
        sync()
        single.append((time.perf_counter() - start) * 1000)
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        predict_images(model, images)
        sync()
        runs.append(time.perf_counter() - start)
    return {
        "single_image": _percentiles(single),
        "batch": {
            "images": len(images),
            "seconds": round(min(runs), 4),
            "images_per_second": round(len(images) / min(runs)),
        },
    }


def _png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart_body(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    """Encode a single-file ``multipart/form-data`` body with the standard library."""
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def time_http(url: str, image: np.ndarray, *, n_requests: int = 200) -> dict[str, Any]:
    """Sequential ``POST /predict`` latency against a running server."""
    body, content_type = multipart_body("file", "sample.png", _png(image))

    def call() -> None:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/predict", data=body, headers={"Content-Type": content_type}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()

    for _ in range(10):
        call()
    samples = []
    start_all = time.perf_counter()
    for _ in range(n_requests):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    total = time.perf_counter() - start_all
    return {
        **_percentiles(samples),
        "requests": n_requests,
        "requests_per_second": round(n_requests / total),
    }


def run_benchmark(settings: Settings, *, url: str | None = None) -> dict[str, Any]:
    """Benchmark the configured model on CPU, on CUDA if available, and optionally over HTTP."""
    _, test = load_arrays(settings.resolve(settings.data_dir))
    images = test.X[:1000]
    model_path = settings.resolve(settings.model_path)
    result: dict[str, Any] = {
        "model": model_path.name,
        "artifact_mb": round(model_path.stat().st_size / 1e6, 2),
        "cpu": time_device(settings, torch.device("cpu"), images),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
        },
    }
    if torch.cuda.is_available():
        result["cuda"] = time_device(settings, torch.device("cuda"), images)
        result["environment"]["gpu"] = torch.cuda.get_device_name(0)
    if url:
        result["http"] = time_http(url, images[0])
    return result
