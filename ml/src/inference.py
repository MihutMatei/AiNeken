"""High-level inference interface for the laptop-price PricingModel.

Single import surface for the API and MCP layers. They never need to know
about joblib, sklearn, or the file layout under ml/artifacts/.

    from ml.src.inference import get_engine

    engine = get_engine()
    engine.predict(features)            # -> PredictionResponse
    engine.explain(features)            # -> FeatureContributionsResponse
    engine.compare(features_a, features_b)  # -> ComparisonResponse
    engine.model_card()                 # -> ModelCard

The engine is constructed once (module-level singleton via `get_engine`).
First call costs ~1s (joblib load + a small read of the dataset for the
baseline calibration); subsequent calls are sub-millisecond.

This module is the contract with the API/MCP teammates. Pydantic return
types are defined in shared/schema.py — changes here propagate to the API
docs and MCP tool definitions automatically.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS
from shared.schema import (
    ComparisonResponse,
    FeatureContribution,
    FeatureContributionsResponse,
    FeatureDelta,
    LaptopFeatures,
    ModelCard,
    PredictionResponse,
)

DEFAULT_ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "ml/artifacts"))
DEFAULT_DATA = Path(os.getenv("DATASET_PATH", "data/laptop_price_dataset.csv"))

FEATURE_ORDER = CATEGORICAL_COLS + NUMERIC_COLS


def _features_to_df(features: LaptopFeatures) -> pd.DataFrame:
    """Pydantic LaptopFeatures -> single-row DataFrame in the column order
    the model was trained on. Extra fields on LaptopFeatures (screen size,
    weight, etc.) are silently ignored — the API can still accept them,
    the model just doesn't use them."""
    row = features.model_dump(exclude={"model_name"}, mode="json")
    keep = {col: row[col] for col in FEATURE_ORDER}
    return pd.DataFrame([keep])[FEATURE_ORDER]


class InferenceEngine:
    """Wraps the trained PricingModel + its model card.

    The PricingModel itself carries all the introspection methods we need
    (`contributions`, `baseline_usd`, `card_payload`) — this class is a
    thin Pydantic-typed adapter over it.
    """

    def __init__(
        self,
        artifacts_dir: Path = DEFAULT_ARTIFACTS,
        data_path: Path = DEFAULT_DATA,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.data_path = Path(data_path)

        self.model = joblib.load(self.artifacts_dir / "model.pkl")
        self.card: dict[str, Any] = json.loads(
            (self.artifacts_dir / "model_card.json").read_text()
        )
        self.version: str = self.card.get("version", "unknown")

    # ------------------------------------------------------------------
    # Primitive 1: predict
    # ------------------------------------------------------------------

    def predict(self, features: LaptopFeatures) -> PredictionResponse:
        df = _features_to_df(features)
        price = float(self.model.predict(df)[0])
        return PredictionResponse(
            predicted_price_usd=round(price, 2),
            model_version=self.version,
        )

    # ------------------------------------------------------------------
    # Primitive 2: explain
    # ------------------------------------------------------------------

    def explain(self, features: LaptopFeatures) -> FeatureContributionsResponse:
        df = _features_to_df(features)
        breakdown = self.model.contributions(df)
        baseline = self.model.baseline_usd()
        prediction = float(self.model.predict(df)[0])

        items = [
            FeatureContribution(
                feature=feature_name,
                value=parts["value"],
                shap_usd=parts["total_usd"],
            )
            for feature_name, parts in breakdown.items()
        ]
        items.sort(key=lambda c: abs(c.shap_usd), reverse=True)

        return FeatureContributionsResponse(
            predicted_price_usd=round(prediction, 2),
            baseline_usd=round(baseline, 2),
            contributions=items,
            model_version=self.version,
        )

    # ------------------------------------------------------------------
    # Primitive 3: compare
    # ------------------------------------------------------------------

    def compare(
        self,
        features_a: LaptopFeatures,
        features_b: LaptopFeatures,
    ) -> ComparisonResponse:
        df_a = _features_to_df(features_a)
        df_b = _features_to_df(features_b)
        price_a = float(self.model.predict(df_a)[0])
        price_b = float(self.model.predict(df_b)[0])

        # Per-feature delta: swap each differing column from A to B,
        # measure the price change. O(n_features) predictions — cheap.
        deltas: list[FeatureDelta] = []
        row_a = df_a.iloc[0]
        row_b = df_b.iloc[0]
        for col in FEATURE_ORDER:
            if row_a[col] == row_b[col]:
                continue
            swapped = df_a.copy()
            swapped[col] = row_b[col]
            new_price = float(self.model.predict(swapped)[0])
            deltas.append(
                FeatureDelta(
                    feature=col,
                    value_a=row_a[col].item() if hasattr(row_a[col], "item") else row_a[col],
                    value_b=row_b[col].item() if hasattr(row_b[col], "item") else row_b[col],
                    delta_usd=round(new_price - price_a, 2),
                )
            )
        deltas.sort(key=lambda d: abs(d.delta_usd), reverse=True)

        return ComparisonResponse(
            price_a_usd=round(price_a, 2),
            price_b_usd=round(price_b, 2),
            delta_usd=round(price_b - price_a, 2),
            feature_deltas=deltas,
            model_version=self.version,
        )

    # ------------------------------------------------------------------
    # Primitive 4: model_card
    # ------------------------------------------------------------------

    def model_card(self) -> ModelCard:
        return ModelCard.model_validate(self.card)


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------

_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()


def get_engine(
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS,
    data_path: Path | str = DEFAULT_DATA,
) -> InferenceEngine:
    """Lazy, thread-safe singleton. The kwargs are honored only on the first
    call (when the engine is first constructed). For tests that want a fresh
    engine, set ml.src.inference._engine = None and call again."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InferenceEngine(Path(artifacts_dir), Path(data_path))
    return _engine
