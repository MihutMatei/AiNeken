"""Re-score a saved model on the dataset (or a held-out CSV).

Usage:
    python -m ml.src.evaluate --model ml/artifacts/model.pkl --data data/laptop_price_dataset.csv
"""

from __future__ import annotations

import argparse
import json

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.preprocess import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    X, y = load_dataset(args.data)
    preds = pipe.predict(X)

    print(json.dumps({
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(mean_squared_error(y, preds) ** 0.5),
        "r2": float(r2_score(y, preds)),
    }, indent=2))


if __name__ == "__main__":
    main()
