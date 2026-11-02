"""FastAPI inference service: upload images, get CIFAR-10 class predictions."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse

from cifar_classifier import __version__
from cifar_classifier.config import Settings, get_settings
from cifar_classifier.engine import resolve_device
from cifar_classifier.predict import (
    ImageDecodeError,
    InferenceModel,
    ModelLoadError,
    ModelNotFoundError,
    decode_image,
    load_model,
    predict_images,
)
from cifar_classifier.schema import BatchResponse, HealthResponse, NamedPrediction, Prediction

logger = logging.getLogger(__name__)

Loader = Callable[[Path, torch.device], InferenceModel]


def create_app(settings: Settings | None = None, loader: Loader = load_model) -> FastAPI:
    """Build the application.

    The model is loaded once at startup. If that fails the service still starts so
    ``/health`` can report it, and prediction endpoints return 503.

    Args:
        settings: Runtime settings; defaults to environment-derived settings.
        loader: Function that loads the model (injectable for tests).

    Returns:
        The configured FastAPI app.
    """
    settings = settings or get_settings()
    device = resolve_device(settings.device)
    model_path = settings.resolve(settings.model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = None
        try:
            app.state.model = loader(model_path, device)
            logger.info("Loaded %s on %s", model_path.name, device)
        except (ModelNotFoundError, ModelLoadError) as exc:
            logger.error("Model not loaded: %s", exc)
        yield

    app = FastAPI(
        title="CIFAR-10 Image Classifier",
        version=__version__,
        description="Upload images (resized to 32x32) and get one of the ten CIFAR-10 classes.",
        lifespan=lifespan,
    )

    def get_model(request: Request) -> InferenceModel:
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
        return model

    async def read_image(upload: UploadFile) -> np.ndarray:
        data = await upload.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"{upload.filename} exceeds {settings.max_upload_bytes} bytes",
            )
        try:
            return decode_image(data)
        except ImageDecodeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"{upload.filename}: {exc}"
            ) from exc

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health(request: Request) -> HealthResponse:
        """Liveness, plus whether the model is ready to serve."""
        loaded = getattr(request.app.state, "model", None) is not None
        return HealthResponse(status="ok", model_loaded=loaded, device=str(device))

    @app.get("/model/info")
    def model_info(request: Request) -> dict[str, Any]:
        """Training metadata of the loaded model."""
        return get_model(request).metadata

    @app.post("/predict")
    async def predict(
        request: Request, file: Annotated[UploadFile, File(description="PNG/JPEG image")]
    ) -> Prediction:
        """Classify one image."""
        model = get_model(request)
        return predict_images(model, (await read_image(file))[None])[0]

    @app.post("/predict/batch")
    async def predict_batch(
        request: Request, files: Annotated[list[UploadFile], File(description="Images")]
    ) -> BatchResponse:
        """Classify up to ``CIFAR_MAX_BATCH_SIZE`` images in one call."""
        if len(files) > settings.max_batch_size:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Batch of {len(files)} exceeds the limit of {settings.max_batch_size}",
            )
        model = get_model(request)
        images = np.stack([await read_image(f) for f in files])
        predictions = predict_images(model, images)
        return BatchResponse(
            predictions=[
                NamedPrediction(**p.model_dump(), filename=f.filename)
                for p, f in zip(predictions, files, strict=True)
            ]
        )

    return app
