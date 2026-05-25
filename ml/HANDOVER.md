# ML → API/MCP handover

Status: **trained model committed** to `ml/artifacts/`. The API and MCP layers can integrate immediately, no training required on your side.

---

## TL;DR — how you call the model

```python
from ml.src.inference import get_engine
from shared.schema import LaptopFeatures

engine = get_engine()          # lazy singleton, warm on first call (~1s)

engine.predict(features)       # → PredictionResponse
engine.explain(features)       # → FeatureContributionsResponse  (per-feature SHAP)
engine.compare(a, b)           # → ComparisonResponse           (diff two configs)
engine.model_card()            # → ModelCard                    (version, metrics, schema)
```

That's the entire surface. All four return Pydantic models defined in `shared/schema.py` — you can return them straight from FastAPI without re-validating.

---

## What's in `ml/artifacts/` (committed to git)

| File | Purpose |
|---|---|
| `model.pkl` | Trained sklearn `Pipeline` (preprocessor + `GradientBoostingRegressor`). 146 KB. |
| `model_card.json` | Version, estimator, split sizes, val + test metrics, feature schema. |
| `split.json` | Frozen train/val/test row indices (seed=42). Required by `inference.py`. |

Pull and you're ready to integrate — no `pip install -r requirements.txt` for ML needed unless you want to retrain.

---

## What changed since the scaffold

| Area | Change | Why it matters to you |
|---|---|---|
| `shared/schema.py` | Added `FeatureContribution(s)Response`, `ComparisonResponse`, `FeatureDelta`, `ModelCard`. `LaptopFeatures` unchanged. | New response types are ready to wire into FastAPI routes and MCP tool return types. |
| `ml/src/inference.py` | **New.** Single import surface (`get_engine()`) for all model interactions. | Use this — don't reach into `joblib.load(...)` directly. |
| `ml/src/split.py` | **New.** 3-way stratified split, persisted to `split.json`. | `inference.py` reads `split.json` to compute the SHAP baseline. Don't delete the file. |
| `ml/src/train.py` | Reads `split.json`. Fits on train, scores on val, leaves test sealed. | If you retrain, val metrics auto-update in `model_card.json`. Test metrics stay until you run `evaluate --split test --finalize`. |
| `ml/src/evaluate.py` | Adds `--split {train,val,test}` and a `--finalize` gate for the sealed test pass. | You probably won't need this — just useful if you want to sanity-check after a retrain. |
| `ml/src/preprocess.py` | Coerces numeric columns and drops corrupt rows (the CSV has 2 typos like `3test`). Quiet `[preprocess] dropped N row(s)` line on load. | Expect that log line when you import the engine — it's not an error. |
| `requirements.txt` | Added `shap`, `numpy`, `pytest`. | Already installed in `.venv/` if you `pip install -r requirements.txt`. |

---

## How to swap the existing API stub for the real engine

`api/app/model_loader.py` currently joblib-loads the model and exposes `pipeline.predict`. Replace that with the engine:

```python
# api/app/model_loader.py — proposed
from ml.src.inference import get_engine

def get_inference_engine():
    return get_engine()
```

Then in `api/app/main.py`:

```python
from api.app.model_loader import get_inference_engine
from shared.schema import (
    ComparisonResponse,
    FeatureContributionsResponse,
    HealthResponse,
    LaptopFeatures,
    ModelCard,
    PredictionResponse,
)

engine = get_inference_engine()

@app.post("/predict", response_model=PredictionResponse)
def predict(features: LaptopFeatures):
    return engine.predict(features)

@app.post("/explain", response_model=FeatureContributionsResponse)
def explain(features: LaptopFeatures):
    return engine.explain(features)

@app.post("/compare", response_model=ComparisonResponse)
def compare(payload: dict):              # body = {"a": LaptopFeatures, "b": LaptopFeatures}
    a = LaptopFeatures(**payload["a"])
    b = LaptopFeatures(**payload["b"])
    return engine.compare(a, b)

@app.get("/model-card", response_model=ModelCard)
def model_card():
    return engine.model_card()

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=True, model_version=engine.version)
```

For the MCP server in `mcp_server/server.py`, each FastAPI route becomes one `@mcp.tool()`. The richer demo prompt the panel wants:

> *"Compare a Dell with 16GB vs 32GB RAM, explain the price drivers, tell me whether the cheaper one is the better value."*

…requires the LLM to chain `predict` + `compare` + `explain`. Expose all three.

---

## One sharp edge worth knowing

`get_engine()` builds a singleton on first call. It reads `data/laptop_price_dataset.csv` to compute the SHAP baseline (mean prediction over the train split). So:

- The CSV needs to be present at the path in `DATASET_PATH` (default `data/laptop_price_dataset.csv`).
- First call takes ~1s; subsequent calls are sub-millisecond.
- For the container build, copy `data/` into the image (already done in `mcp_server/Dockerfile`; do the same in `api/Dockerfile` if it's missing).

---

## Honesty disclaimer (for the demo)

The dataset has essentially **no signal** — predicted prices land near the mean ($1780) regardless of inputs, and R² hovers around 0. Per Matei's call, we ship as-is; the brief grades the pipeline (data → ML → API → MCP → LLM), not regression accuracy. Don't oversell the model in the demo — frame it as "the model gives a number and an explanation, here's how the LLM consumes them."
