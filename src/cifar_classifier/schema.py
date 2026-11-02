"""API response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    """Probability of one class."""

    class_name: str
    probability: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    """Prediction for one image."""

    label: int = Field(ge=0, le=9, description="Predicted class index.")
    class_name: str
    probability: float = Field(ge=0, le=1, description="Probability of the predicted class.")
    top_k: list[ClassScore] = Field(description="The three most likely classes.")


class NamedPrediction(Prediction):
    """Prediction tagged with the uploaded file name."""

    filename: str | None = None


class BatchResponse(BaseModel):
    """Predictions in upload order."""

    predictions: list[NamedPrediction]


class HealthResponse(BaseModel):
    """Service liveness and readiness."""

    status: str
    model_loaded: bool
    device: str
