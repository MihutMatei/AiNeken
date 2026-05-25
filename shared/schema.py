"""Shared contract between ML, API and MCP server.

Edit with care: any change here ripples through training, inference,
and the MCP tool signature. Keep field names in sync with the CSV.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Brand(str, Enum):
    DELL = "Dell"
    ASUS = "Asus"
    HP = "HP"
    ACER = "Acer"
    LENOVO = "Lenovo"
    MSI = "MSI"
    APPLE = "Apple"
    RAZER = "Razer"


class Processor(str, Enum):
    INTEL_I5 = "Intel i5"
    INTEL_I7 = "Intel i7"
    INTEL_I9 = "Intel i9"
    RYZEN_5 = "AMD Ryzen 5"
    RYZEN_7 = "AMD Ryzen 7"
    RYZEN_9 = "AMD Ryzen 9"


class GraphicsCard(str, Enum):
    INTEL_UHD = "Intel UHD"
    AMD_RADEON = "AMD Radeon"
    GTX_1650 = "NVIDIA GTX 1650"
    RTX_3060 = "NVIDIA RTX 3060"
    RTX_3070 = "NVIDIA RTX 3070"


class OperatingSystem(str, Enum):
    WIN_10 = "Windows 10"
    WIN_11 = "Windows 11"
    LINUX = "Linux"
    MACOS = "macOS"


class LaptopFeatures(BaseModel):
    """Single laptop record. Field names mirror the CSV columns (snake_case)."""

    brand: Brand
    processor: Processor
    ram_gb: int = Field(..., ge=2, le=256, description="RAM in GB")
    storage_gb: int = Field(..., ge=64, le=8192, description="Storage in GB")
    screen_size_inches: float = Field(..., ge=10.0, le=20.0)
    graphics_card: GraphicsCard
    operating_system: OperatingSystem
    weight_kg: float = Field(..., gt=0.5, le=5.0)
    battery_life_hours: float = Field(..., ge=1.0, le=30.0)
    warranty_years: int = Field(..., ge=0, le=5)
    # Model name is high-cardinality (3000 unique) — excluded from features by default.
    model_name: str | None = Field(default=None, alias="model")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    currency: Literal["USD"] = "USD"
    model_version: str
    # Optional confidence band — fill if the regressor supports it.
    lower_bound_usd: float | None = None
    upper_bound_usd: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


# --- Richer interrogation responses ---------------------------------------


class FeatureContribution(BaseModel):
    """One feature's contribution to a single prediction (SHAP-style)."""

    feature: str
    value: float | int | str
    shap_usd: float = Field(..., description="Signed USD contribution vs. baseline")


class FeatureContributionsResponse(BaseModel):
    predicted_price_usd: float
    baseline_usd: float = Field(
        ..., description="Mean predicted price across the training set"
    )
    contributions: list[FeatureContribution] = Field(
        ..., description="Sorted by absolute USD impact, descending"
    )
    model_version: str


class FeatureDelta(BaseModel):
    feature: str
    value_a: float | int | str
    value_b: float | int | str
    delta_usd: float = Field(..., description="Signed USD impact of moving from A to B")


class ComparisonResponse(BaseModel):
    price_a_usd: float
    price_b_usd: float
    delta_usd: float = Field(..., description="price_b - price_a")
    feature_deltas: list[FeatureDelta] = Field(
        ..., description="Only features that differ between A and B, sorted by |delta|"
    )
    model_version: str


class FeatureSchema(BaseModel):
    categorical: list[str]
    numeric: list[str]


class ModelCard(BaseModel):
    version: str
    trained_at: str
    estimator: str
    split_seed: int | None = None
    n_train: int
    n_val: int
    n_test: int
    val_metric_mae: float | None = None
    val_metric_rmse: float | None = None
    val_metric_r2: float | None = None
    test_metric_mae: float | None = None
    test_metric_rmse: float | None = None
    test_metric_r2: float | None = None
    feature_columns: FeatureSchema
    target: str
