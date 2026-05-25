"""Three-way stratified split: train / val / test.

Run once at the start of the ML session. Persists row indices to split.json so
every later artifact (model, quantile companions, SHAP explainer, evaluation)
uses the exact same partition.

Usage:
    python -m ml.src.split --data data/laptop_price_dataset.csv --out ml/artifacts/

Default proportions: 70% train / 15% val / 15% test.
Stratified by price quintile so all three splits cover the same price bands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.src.preprocess import load_dataset

SEED = 42
N_PRICE_BINS = 5


def make_splits(
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Two-step stratified split. Returns positional row indices for each set."""
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must each be in (0,1) and sum to < 1")

    strata = pd.qcut(y, q=N_PRICE_BINS, duplicates="drop").cat.codes.to_numpy()
    all_idx = list(range(len(y)))

    # Peel off the test set first — it gets sealed and never looked at again.
    rest_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=strata,
        random_state=seed,
    )

    # Split the remainder into train + val. val_size is expressed relative to
    # the original total, so rescale to the remainder.
    val_share_of_rest = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=val_share_of_rest,
        stratify=strata[rest_idx],
        random_state=seed,
    )

    return {
        "seed": seed,
        "n_total": len(y),
        "train_idx": sorted(train_idx),
        "val_idx": sorted(val_idx),
        "test_idx": sorted(test_idx),
    }


def load_splits(artifacts_dir: str | Path) -> dict[str, list[int]]:
    """Read split.json back. Use this from train.py / evaluate.py."""
    path = Path(artifacts_dir) / "split.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m ml.src.split` first."
        )
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    _, y = load_dataset(args.data)
    splits = make_splits(y, val_size=args.val_size, test_size=args.test_size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(splits))

    print(
        f"split.json written: "
        f"train={len(splits['train_idx'])}, "
        f"val={len(splits['val_idx'])}, "
        f"test={len(splits['test_idx'])} (seed={splits['seed']})"
    )


if __name__ == "__main__":
    main()
