"""Load the trained pipeline from disk, or fall back to a stub so the API
remains usable while Part 1 is still in flight."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = "ml/artifacts/model.pkl"
DEFAULT_CARD_PATH = "ml/artifacts/model_card.json"


@dataclass
class LoadedModel:
    predict_fn: callable
    version: str
    is_stub: bool


def _stub_predict(df: pd.DataFrame) -> list[float]:
    # Crude heuristic so the API returns something plausible until the real
    # model is trained. Replace by deleting nothing — this is only used when
    # MODEL_PATH does not exist.
    base = 500.0
    per_ram = 25.0
    per_storage = 0.4
    return [
        base
        + per_ram * float(row["ram_gb"])
        + per_storage * float(row["storage_gb"])
        + 80.0 * float(row["warranty_years"])
        for _, row in df.iterrows()
    ]


def load_model() -> LoadedModel:
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        return LoadedModel(predict_fn=_stub_predict, version="stub-0.0.0", is_stub=True)

    pipe = joblib.load(path)
    card_path = Path(os.getenv("MODEL_CARD_PATH", DEFAULT_CARD_PATH))
    version = "unknown"
    if card_path.exists():
        try:
            version = json.loads(card_path.read_text()).get("version", "unknown")
        except json.JSONDecodeError:
            pass
    return LoadedModel(predict_fn=pipe.predict, version=version, is_stub=False)
