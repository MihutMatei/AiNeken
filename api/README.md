# Part 2 — FastAPI inference service

**Owner:** _assign one teammate_
**Goal:** serve the trained model behind a clean HTTP API.

## Endpoints
- `GET /health` → `{status, model_loaded, model_version}`
- `POST /predict` → body is `LaptopFeatures`, response is `PredictionResponse` (see `shared/schema.py`).

## Run locally
```bash
# from repo root
export MODEL_PATH=ml/artifacts/model.pkl
uvicorn api.app.main:app --reload --port 8000
```

If `MODEL_PATH` does not exist, the loader falls back to a deterministic
**stub model** so the API is usable before ML lands. `/health` reports
`status: "degraded"` in that case.

## Docker
```bash
docker build -t laptop-api -f api/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/ml/artifacts:/app/ml/artifacts laptop-api
```
