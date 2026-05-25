"""Train baseline regressor for laptop price.

Usage:
    python -m ml.src.train --data data/laptop_price_dataset.csv --out ml/artifacts/

Produces:
    <out>/model.pkl
    <out>/model_card.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS, load_dataset

VERSION = "0.1.0"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    return Pipeline(
        [
            ("preprocess", pre),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    card = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "metric_mae": float(mean_absolute_error(y_te, preds)),
        "metric_rmse": float(mean_squared_error(y_te, preds) ** 0.5),
        "metric_r2": float(r2_score(y_te, preds)),
    }

    joblib.dump(pipe, out / "model.pkl")
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
