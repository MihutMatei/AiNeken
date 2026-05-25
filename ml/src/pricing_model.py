"""Hybrid pricing estimator: hand-picked prior + Ridge residual.

Architecture:

    prediction(x) = prior(x)  +  ridge_residual(x)

where:

    prior(x) = cpu_price[x.processor]
             + gpu_price[x.graphics_card]
             + per_gb_ram * x.ram_gb
             + per_gb_storage * x.storage_gb
             + brand_markup[x.brand]
             + os_premium[x.operating_system]
             + assembly_base

All those numbers live in `ml/src/pricing.py` and are *frozen onto the
estimator instance at fit-time* — so the persisted .pkl is self-contained
and edits to pricing.py after training do not silently change predictions.

Ridge then learns a small correction on top: what the priors miss in the
training data. With the dataset's near-zero signal, the residual is small
and the prediction stays dominated by the priors (which is what we want
for interpretability).

Compatible with the existing inference layer:

    PricingModel().fit(X, y).predict(X)
    PricingModel().contributions(X_row)  -> dict[feature, usd]
    PricingModel().baseline_usd()        -> float

The instance exposes the *effective* (prior + learned residual) per-category
markups for the model card — that's the headline number the demo shows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from ml.src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS
from ml.src.pricing import (
    ASSEMBLY_BASE_USD,
    BRAND_MARKUP_USD,
    CPU_PRICE_USD,
    GPU_PRICE_USD,
    OS_PREMIUM_USD,
    PER_GB_RAM_USD,
    PER_GB_STORAGE_USD,
    prior_price_usd,
)


class PricingModel(BaseEstimator, RegressorMixin):
    """Hand-picked prior + Ridge residual regressor.

    Parameters
    ----------
    alpha
        Ridge regularization strength. Higher = trust the prior more,
        let the data move predictions less. Default 1.0 is gentle.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def _prior(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                prior_price_usd(
                    brand=row["brand"],
                    processor=row["processor"],
                    graphics_card=row["graphics_card"],
                    operating_system=row["operating_system"],
                    ram_gb=float(row["ram_gb"]),
                    storage_gb=float(row["storage_gb"]),
                )
                for _, row in X.iterrows()
            ],
            dtype=float,
        )

    def _build_encoder(self) -> ColumnTransformer:
        return ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_COLS,
                ),
                ("num", "passthrough", NUMERIC_COLS),
            ]
        )

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y) -> "PricingModel":
        y = np.asarray(y, dtype=float)
        prior = self._prior(X)
        residual = y - prior

        self.encoder_ = self._build_encoder()
        Xt = self.encoder_.fit_transform(X)
        self.feature_names_out_ = list(self.encoder_.get_feature_names_out())

        self.ridge_ = Ridge(alpha=self.alpha, random_state=42)
        self.ridge_.fit(Xt, residual)

        # Freeze tables onto the instance so the persisted .pkl carries the
        # numbers it was trained with. Editing ml/src/pricing.py after the
        # model is saved does NOT silently change loaded-model behavior —
        # you have to retrain. This makes saved models reproducible.
        self.frozen_cpu_price_usd_ = dict(CPU_PRICE_USD)
        self.frozen_gpu_price_usd_ = dict(GPU_PRICE_USD)
        self.frozen_brand_markup_usd_ = dict(BRAND_MARKUP_USD)
        self.frozen_os_premium_usd_ = dict(OS_PREMIUM_USD)
        self.frozen_per_gb_ram_usd_ = float(PER_GB_RAM_USD)
        self.frozen_per_gb_storage_usd_ = float(PER_GB_STORAGE_USD)
        self.frozen_assembly_base_usd_ = float(ASSEMBLY_BASE_USD)

        # Effective (prior + learned residual) markups — the "what the model
        # actually thinks" numbers, dollar-denominated, ready for the card.
        self.effective_brand_markup_usd_ = self._effective_per_category(
            "brand", BRAND_MARKUP_USD
        )
        self.effective_os_premium_usd_ = self._effective_per_category(
            "operating_system", OS_PREMIUM_USD
        )
        self.effective_cpu_price_usd_ = self._effective_per_category(
            "processor", CPU_PRICE_USD
        )
        self.effective_gpu_price_usd_ = self._effective_per_category(
            "graphics_card", GPU_PRICE_USD
        )
        self.effective_per_gb_ram_usd_ = round(
            self.frozen_per_gb_ram_usd_ + self._coef_for("num__ram_gb"), 4
        )
        self.effective_per_gb_storage_usd_ = round(
            self.frozen_per_gb_storage_usd_ + self._coef_for("num__storage_gb"), 4
        )
        self.ridge_intercept_usd_ = float(self.ridge_.intercept_)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prior = self._prior(X)
        Xt = self.encoder_.transform(X)
        return prior + self.ridge_.predict(Xt)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _coef_for(self, transformed_name: str) -> float:
        """Ridge coefficient on the named transformed feature, or 0 if absent."""
        try:
            i = self.feature_names_out_.index(transformed_name)
            return float(self.ridge_.coef_[i])
        except ValueError:
            return 0.0

    def _effective_per_category(
        self, column: str, prior_table: dict[str, float]
    ) -> dict[str, float]:
        """For a one-hot encoded column, return prior + learned residual per value."""
        return {
            value: round(prior_val + self._coef_for(f"cat__{column}_{value}"), 2)
            for value, prior_val in prior_table.items()
        }

    # ------------------------------------------------------------------
    # Per-row breakdown — used by inference.explain()
    # ------------------------------------------------------------------

    def contributions(self, X_row: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Per-feature USD contribution for a single row.

        Returns a dict of {feature_name: {value, prior_usd, residual_usd, total_usd}}.
        Sum of `total_usd` + `baseline_usd()` == `predict(X_row)[0]`.
        """
        row = X_row.iloc[0]

        def cat_breakdown(col: str, prior_table: dict[str, float]) -> dict[str, Any]:
            value = row[col]
            prior_val = float(prior_table.get(value, 0.0))
            residual = self._coef_for(f"cat__{col}_{value}")
            return {
                "value": value,
                "prior_usd": round(prior_val, 2),
                "residual_usd": round(residual, 2),
                "total_usd": round(prior_val + residual, 2),
            }

        def num_breakdown(col: str, per_unit_prior: float) -> dict[str, Any]:
            value = float(row[col])
            prior_val = per_unit_prior * value
            residual = self._coef_for(f"num__{col}") * value
            return {
                "value": value,
                "prior_usd": round(prior_val, 2),
                "residual_usd": round(residual, 2),
                "total_usd": round(prior_val + residual, 2),
            }

        return {
            "brand": cat_breakdown("brand", self.frozen_brand_markup_usd_),
            "processor": cat_breakdown("processor", self.frozen_cpu_price_usd_),
            "graphics_card": cat_breakdown(
                "graphics_card", self.frozen_gpu_price_usd_
            ),
            "operating_system": cat_breakdown(
                "operating_system", self.frozen_os_premium_usd_
            ),
            "ram_gb": num_breakdown("ram_gb", self.frozen_per_gb_ram_usd_),
            "storage_gb": num_breakdown(
                "storage_gb", self.frozen_per_gb_storage_usd_
            ),
        }

    def baseline_usd(self) -> float:
        """Constant floor — assembly base + Ridge intercept.

        Everyone gets this regardless of features. The breakdown from
        `contributions()` plus this baseline equals the predicted price.
        """
        return round(self.frozen_assembly_base_usd_ + self.ridge_intercept_usd_, 2)

    # ------------------------------------------------------------------
    # Card summary
    # ------------------------------------------------------------------

    def card_payload(self) -> dict[str, Any]:
        """Slice of state that should land in model_card.json."""
        return {
            "alpha": self.alpha,
            "assembly_base_usd": self.frozen_assembly_base_usd_,
            "ridge_intercept_usd": round(self.ridge_intercept_usd_, 2),
            "per_gb_ram_usd": {
                "prior": self.frozen_per_gb_ram_usd_,
                "effective": self.effective_per_gb_ram_usd_,
            },
            "per_gb_storage_usd": {
                "prior": self.frozen_per_gb_storage_usd_,
                "effective": self.effective_per_gb_storage_usd_,
            },
            "brand_markup_usd": {
                "prior": self.frozen_brand_markup_usd_,
                "effective": self.effective_brand_markup_usd_,
            },
            "os_premium_usd": {
                "prior": self.frozen_os_premium_usd_,
                "effective": self.effective_os_premium_usd_,
            },
            "cpu_price_usd": {
                "prior": self.frozen_cpu_price_usd_,
                "effective": self.effective_cpu_price_usd_,
            },
            "gpu_price_usd": {
                "prior": self.frozen_gpu_price_usd_,
                "effective": self.effective_gpu_price_usd_,
            },
        }
