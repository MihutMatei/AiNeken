"""Load the CSV and return a feature DataFrame + target Series.

The CSV column names (with spaces and units) are normalized to snake_case here,
once. After this module, no code should reference the original CSV headers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.csv_columns import CSV_TO_FIELD, DROP_FROM_FEATURES, TARGET


CATEGORICAL_COLS = ["brand", "processor", "graphics_card", "operating_system"]
NUMERIC_COLS = [
    "ram_gb",
    "storage_gb",
    "screen_size_inches",
    "weight_kg",
    "battery_life_hours",
    "warranty_years",
]


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path).rename(columns=CSV_TO_FIELD)
    y = df[TARGET]
    X = df.drop(columns=[TARGET, *DROP_FROM_FEATURES])
    assert set(X.columns) == set(CATEGORICAL_COLS + NUMERIC_COLS), (
        f"Feature columns drifted: {set(X.columns)}"
    )
    return X, y
