"""Load the CSV and return a feature DataFrame + target Series.

The CSV column names (with spaces and units) are normalized to snake_case here,
once. After this module, no code should reference the original CSV headers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.csv_columns import CSV_TO_FIELD, DROP_FROM_FEATURES, TARGET


# Only these 6 columns matter for pricing. The schema still accepts the
# others (screen size, weight, battery life, warranty) so the API contract
# is unchanged — the model just ignores them.
CATEGORICAL_COLS = ["brand", "processor", "graphics_card", "operating_system"]
NUMERIC_COLS = ["ram_gb", "storage_gb"]
IGNORED_COLS = ["screen_size_inches", "weight_kg", "battery_life_hours", "warranty_years"]


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    # `on_bad_lines='skip'` skips rows with the wrong number of fields
    # (e.g. trailing commas, embedded newlines). Numeric coercion below
    # handles cell-level corruption like "3test" or "1UHD737.73".
    df = pd.read_csv(csv_path, on_bad_lines="skip").rename(columns=CSV_TO_FIELD)

    for col in NUMERIC_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=NUMERIC_COLS + [TARGET]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        # Quiet print rather than logging — this runs in scripts.
        print(f"[preprocess] dropped {dropped} row(s) with non-numeric values.")

    y = df[TARGET]
    X = df.drop(columns=[TARGET, *DROP_FROM_FEATURES, *IGNORED_COLS])
    assert set(X.columns) == set(CATEGORICAL_COLS + NUMERIC_COLS), (
        f"Feature columns drifted: got {set(X.columns)}, "
        f"expected {set(CATEGORICAL_COLS + NUMERIC_COLS)}"
    )
    return X, y
