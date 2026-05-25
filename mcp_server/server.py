"""FastMCP server wrapping the FastAPI predictor as an MCP tool.

Run modes:
    python -m mcp_server.server          # stdio (Claude Desktop / Claude Code)
    python -m mcp_server.server --http   # streamable HTTP (OpenWebUI)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from shared.schema import LaptopFeatures, PredictionResponse

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATASET_PATH = os.getenv("DATASET_PATH", "data/laptop_price_dataset.csv")

mcp = FastMCP("laptop-price-predictor")


@mcp.tool()
def predict_laptop_price(features: LaptopFeatures) -> PredictionResponse:
    """Predict the market price (USD) of a laptop from its specs."""
    payload = features.model_dump(mode="json", by_alias=True)
    resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=30.0)
    resp.raise_for_status()
    return PredictionResponse.model_validate(resp.json())


@mcp.resource("dataset://stats")
def dataset_stats() -> str:
    """Summary statistics of the training dataset, as JSON."""
    import pandas as pd

    from shared.csv_columns import CSV_TO_FIELD

    df = pd.read_csv(DATASET_PATH).rename(columns=CSV_TO_FIELD)
    summary = {
        "n_rows": len(df),
        "price_usd": {
            "min": float(df["price_usd"].min()),
            "max": float(df["price_usd"].max()),
            "mean": float(df["price_usd"].mean()),
        },
        "brands": df["brand"].value_counts().to_dict(),
        "processors": df["processor"].value_counts().to_dict(),
        "operating_systems": df["operating_system"].value_counts().to_dict(),
    }
    return json.dumps(summary, indent=2)


@mcp.prompt()
def interpret_prediction(predicted_price_usd: float, brand: str) -> str:
    """Template guiding the LLM to interpret and act on a prediction."""
    return (
        f"The model predicted ${predicted_price_usd:.2f} for this {brand} laptop.\n"
        "1. Compare against the dataset stats (dataset://stats).\n"
        "2. State whether the asking price is below / at / above the predicted band.\n"
        "3. Suggest one positioning angle for the product listing."
    )


def main() -> None:
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
