"""Train the laptop-price regressor on the frozen train split.

Reads row indices from ml/artifacts/split.json — so the train/val/test
partition is identical across every experiment, person, and re-run.

The test split is NOT touched here. Final test metrics are written later by
`python -m ml.src.evaluate --split test --finalize`.

Usage:
    python -m ml.src.split  --data data/laptop_price_dataset.csv --out ml/artifacts/
    python -m ml.src.train  --data data/laptop_price_dataset.csv --out ml/artifacts/

Produces:
    <out>/model.pkl
    <out>/model_card.json   (val metrics only; test metrics filled in by evaluate.py)
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS, load_dataset
from ml.src.split import load_splits

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


def _score(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    splits = load_splits(out)

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    val_metrics = _score(y_va, pipe.predict(X_va))

    estimator_name = type(pipe.named_steps["model"]).__name__
    card = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "estimator": estimator_name,
        "split_seed": splits.get("seed"),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(splits["test_idx"]),
        "val_metric_mae": val_metrics["mae"],
        "val_metric_rmse": val_metrics["rmse"],
        "val_metric_r2": val_metrics["r2"],
        # Filled in by `python -m ml.src.evaluate --split test --finalize`.
        "test_metric_mae": None,
        "test_metric_rmse": None,
        "test_metric_r2": None,
        "feature_columns": {
            "categorical": CATEGORICAL_COLS,
            "numeric": NUMERIC_COLS,
        },
        "target": "price_usd",
    }

    joblib.dump(pipe, out / "model.pkl")
    (out / "model_card.json").write_text(json.dumps(card, indent=2))

    print(json.dumps({"estimator": estimator_name, "val": val_metrics}, indent=2))
    print(f"\nWrote {out / 'model.pkl'} and {out / 'model_card.json'}.")
    print(
        "Test metrics are intentionally None. "
        "Run `python -m ml.src.evaluate --split test --finalize` ONCE you're done tuning."
    )


if __name__ == "__main__":
    main()
