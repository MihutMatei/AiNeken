"""Train the laptop-price PricingModel on the frozen train split.

Architecture is a hybrid prior + Ridge residual (see ml/src/pricing_model.py).
The hand-picked numbers live in ml/src/pricing.py — edit them and re-run
this script to see the predictions and effective markups move accordingly.

Reads row indices from ml/artifacts/split.json so the train/val/test
partition is identical across every experiment and person.

The test split is NOT touched here. Final test metrics are written later by
`python -m ml.src.evaluate --split test --finalize`.

Usage:
    python -m ml.src.split  --data data/laptop_price_dataset.csv --out ml/artifacts/
    python -m ml.src.train  --data data/laptop_price_dataset.csv --out ml/artifacts/

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS, load_dataset
from ml.src.pricing_model import PricingModel
from ml.src.split import load_splits

VERSION = "0.2.0"
DEFAULT_ALPHA = 1.0


def build_model(alpha: float = DEFAULT_ALPHA) -> PricingModel:
    """Return an unfit PricingModel.

    Kept as a separate function so the API/MCP layers can introspect the
    architecture without having to read train() — and so swapping in a
    different estimator later is a one-liner.
    """
    return PricingModel(alpha=alpha)


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
    ap.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Ridge regularization. Higher = trust the priors more.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    splits = load_splits(out)

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model = build_model(alpha=args.alpha)
    model.fit(X_tr, y_tr)

    train_metrics = _score(y_tr, model.predict(X_tr))
    val_metrics = _score(y_va, model.predict(X_va))

    card = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "estimator": type(model).__name__,
        "architecture": (
            "prior(components + brand_markup + os_premium) + Ridge residual"
        ),
        "split_seed": splits.get("seed"),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(splits["test_idx"]),
        # Training-set metrics are interesting for diagnostics (overfit gap).
        "train_metric_mae": train_metrics["mae"],
        "train_metric_rmse": train_metrics["rmse"],
        "train_metric_r2": train_metrics["r2"],
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
        # The "this is what the model actually thinks" payload — hand-picked
        # priors next to the data-calibrated effective values, all in USD.
        "pricing": model.card_payload(),
    }

    joblib.dump(model, out / "model.pkl")
    (out / "model_card.json").write_text(json.dumps(card, indent=2))

    print("=" * 60)
    print(f"Trained {type(model).__name__} (alpha={args.alpha})")
    print("=" * 60)
    print(f"  train MAE = ${train_metrics['mae']:.2f}   R² = {train_metrics['r2']:+.3f}")
    print(f"  val   MAE = ${val_metrics['mae']:.2f}   R² = {val_metrics['r2']:+.3f}")
    print()
    print("Effective brand markups (USD, prior + data residual):")
    for brand, eff in sorted(
        model.effective_brand_markup_usd_.items(), key=lambda kv: -kv[1]
    ):
        prior = model.frozen_brand_markup_usd_[brand]
        delta = eff - prior
        print(f"  {brand:8s}  prior=${prior:6.0f}   effective=${eff:7.2f}   Δ=${delta:+7.2f}")
    print()
    print(f"Wrote {out / 'model.pkl'} and {out / 'model_card.json'}.")
    print(
        "Test metrics are intentionally None. "
        "Run `python -m ml.src.evaluate --split test --finalize` ONCE you're done tuning."
    )


if __name__ == "__main__":
    main()
