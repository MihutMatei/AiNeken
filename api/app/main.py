"""FastAPI inference service. Loads model on startup, serves /predict."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI

from api.app.model_loader import load_model
from shared.schema import HealthResponse, LaptopFeatures, PredictionResponse

app = FastAPI(title="Laptop Price Predictor", version="0.1.0")
_loaded = load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if not _loaded.is_stub else "degraded",
        model_loaded=not _loaded.is_stub,
        model_version=_loaded.version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: LaptopFeatures) -> PredictionResponse:
    row = features.model_dump(exclude={"model_name"}, mode="json")
    # Enum fields serialize to their string value via mode="json".
    df = pd.DataFrame([row])
    price = float(_loaded.predict_fn(df)[0])
    return PredictionResponse(
        predicted_price_usd=round(price, 2),
        model_version=_loaded.version,
    )
