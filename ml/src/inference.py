"""High-level inference interface for the laptop-price model.

Single import surface for the API and MCP layers. They never need to know
about joblib, SHAP, or the file layout under ml/artifacts/.

    from ml.src.inference import (
        InferenceEngine,
        get_engine,
    )

    engine = get_engine()
    engine.predict(features)
    engine.explain(features)
    engine.compare(features_a, features_b)
    engine.model_card()

The engine is constructed once (module-level singleton via `get_engine`) and
holds the loaded sklearn Pipeline, the SHAP explainer, the model card, and
the training-set mean (used as the SHAP baseline).
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from ml.src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS, load_dataset
from ml.src.split import load_splits
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
    the pipeline was trained on."""
    row = features.model_dump(exclude={"model_name"}, mode="json")
    return pd.DataFrame([row])[FEATURE_ORDER]


class InferenceEngine:
    """Wraps the trained model + SHAP explainer + model card."""

    def __init__(
        self,
        artifacts_dir: Path = DEFAULT_ARTIFACTS,
        data_path: Path = DEFAULT_DATA,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.data_path = Path(data_path)

        self.pipeline = joblib.load(self.artifacts_dir / "model.pkl")
        self.card: dict[str, Any] = json.loads(
            (self.artifacts_dir / "model_card.json").read_text()
        )
        self.version: str = self.card.get("version", "unknown")

        # Baseline = mean prediction on the training split. Used so the
        # contributions add up to (prediction - baseline).
        X, _ = load_dataset(self.data_path)
        splits = load_splits(self.artifacts_dir)
        self._X_train = X.iloc[splits["train_idx"]]
        self.baseline: float = float(self.pipeline.predict(self._X_train).mean())

        # SHAP TreeExplainer on the *fitted estimator*, evaluated on the
        # *transformed* features. We build a small wrapper that applies the
        # preprocessor and then calls the explainer.
        self._preprocessor = self.pipeline.named_steps["preprocess"]
        self._estimator = self.pipeline.named_steps["model"]
        self._feature_names_out = list(self._preprocessor.get_feature_names_out())
        self._explainer = shap.TreeExplainer(self._estimator)

    # --- Primitive 1: predict ------------------------------------------------

    def predict(self, features: LaptopFeatures) -> PredictionResponse:
        df = _features_to_df(features)
        price = float(self.pipeline.predict(df)[0])
        return PredictionResponse(
            predicted_price_usd=round(price, 2),
            model_version=self.version,
        )

    # --- Primitive 2: explain ------------------------------------------------

    def explain(self, features: LaptopFeatures) -> FeatureContributionsResponse:
        df = _features_to_df(features)
        Xt = self._preprocessor.transform(df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        shap_row = np.asarray(self._explainer.shap_values(Xt))[0]

        # Sum up the SHAP values back to the original (pre-one-hot) feature names.
        # OneHotEncoder produces "cat__brand_Dell" etc. so we split off the prefix.
        contributions: dict[str, float] = {name: 0.0 for name in FEATURE_ORDER}
        for transformed_name, shap_val in zip(self._feature_names_out, shap_row):
            base = transformed_name.split("__", 1)[-1]
            original = base.split("_", 1)[0] if base.split("_", 1)[0] in contributions else None
            # Best-effort match: longest prefix that is one of our columns.
            if original is None:
                for col in FEATURE_ORDER:
                    if base.startswith(col):
                        original = col
                        break
            if original is None:
                continue
            contributions[original] += float(shap_val)

        row = df.iloc[0]
        items = [
            FeatureContribution(
                feature=col,
                value=row[col].item() if hasattr(row[col], "item") else row[col],
                shap_usd=round(contributions[col], 2),
            )
            for col in FEATURE_ORDER
        ]
        items.sort(key=lambda c: abs(c.shap_usd), reverse=True)

        return FeatureContributionsResponse(
            predicted_price_usd=round(float(self.pipeline.predict(df)[0]), 2),
            baseline_usd=round(self.baseline, 2),
            contributions=items,
            model_version=self.version,
        )

    # --- Primitive 3: compare ------------------------------------------------

    def compare(
        self,
        features_a: LaptopFeatures,
        features_b: LaptopFeatures,
    ) -> ComparisonResponse:
        df_a = _features_to_df(features_a)
        df_b = _features_to_df(features_b)
        price_a = float(self.pipeline.predict(df_a)[0])
        price_b = float(self.pipeline.predict(df_b)[0])

        # Per-feature delta via "swap one column at a time from A to B" and
        # measure the price change. O(n_features) predictions — cheap.
        deltas: list[FeatureDelta] = []
        row_a = df_a.iloc[0]
        row_b = df_b.iloc[0]
        for col in FEATURE_ORDER:
            if row_a[col] == row_b[col]:
                continue
            swapped = df_a.copy()
            swapped[col] = row_b[col]
            new_price = float(self.pipeline.predict(swapped)[0])
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

    # --- Primitive 4: model_card --------------------------------------------

    def model_card(self) -> ModelCard:
        return ModelCard.model_validate(self.card)


# --- Module-level singleton -------------------------------------------------

_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()


def get_engine(
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS,
    data_path: Path | str = DEFAULT_DATA,
) -> InferenceEngine:
    """Lazy, thread-safe singleton. Re-import in tests with a fresh dir."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InferenceEngine(Path(artifacts_dir), Path(data_path))
    return _engine
