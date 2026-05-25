"""Score a saved model on one of the frozen splits.

Default: re-score on validation (safe, repeatable). Use --split test only
ONCE, at the very end, to fill in the sealed test metrics on model_card.json.

Usage:
    # Default — score on val
    python -m ml.src.evaluate --data data/laptop_price_dataset.csv

    # Score on train (sanity check for over/underfit)
    python -m ml.src.evaluate --data data/laptop_price_dataset.csv --split train

    # Final test pass — writes test metrics into model_card.json
    python -m ml.src.evaluate --data data/laptop_price_dataset.csv --split test --finalize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.preprocess import load_dataset
from ml.src.split import load_splits


def _score(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--artifacts", default="ml/artifacts")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="Required with --split test. Writes the sealed test metrics into model_card.json.",
    )
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    model_path = artifacts / "model.pkl"
    card_path = artifacts / "model_card.json"

    if args.split == "test" and not args.finalize:
        raise SystemExit(
            "Refusing to score on test without --finalize. "
            "The test split is sealed until you're done tuning."
        )

    pipe = joblib.load(model_path)
    X, y = load_dataset(args.data)
    splits = load_splits(artifacts)
    idx = splits[f"{args.split}_idx"]
    X_eval, y_eval = X.iloc[idx], y.iloc[idx]

    metrics = _score(y_eval, pipe.predict(X_eval))
    out = {"split": args.split, "n": len(idx), **metrics}
    print(json.dumps(out, indent=2))

    if args.split == "test" and args.finalize:
        card = json.loads(card_path.read_text())
        card["test_metric_mae"] = metrics["mae"]
        card["test_metric_rmse"] = metrics["rmse"]
        card["test_metric_r2"] = metrics["r2"]
        card_path.write_text(json.dumps(card, indent=2))
        print(f"\nSealed test metrics written to {card_path}.")


if __name__ == "__main__":
    main()
