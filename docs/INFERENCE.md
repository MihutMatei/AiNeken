# Inference — Documentație Tehnică

> Acoperă: `model.pkl`, `InferenceEngine`, endpoint-urile FastAPI, tool-urile MCP și datele tehnice ale modelului.

---

## 1. Model antrenat — `ml/artifacts/model.pkl`

| Proprietate | Valoare |
|---|---|
| Tip | `sklearn.pipeline.Pipeline` |
| Estimator | `GradientBoostingRegressor` |
| Preprocessor | `ColumnTransformer` (OneHotEncoder + StandardScaler) |
| Target | `price_usd` (USD, float) |
| Dimensiune fișier | ~146 KB |
| Seed split | 42 |
| Split | 70% train / 15% val / 15% test |

### Metrici (validare)

Valorile exacte se găsesc în `ml/artifacts/model_card.json`.

| Metrică | Val set |
|---|---|
| MAE | vezi `model_card.json → val_metric_mae` |
| RMSE | vezi `model_card.json → val_metric_rmse` |
| R² | ~0 (dataset fără semnal real — vezi nota de onestitate) |

> **Nota de onestitate:** Dataset-ul nu are semnal predictiv real — prețurile prezise aterizează aproape de medie (~$1780) indiferent de input. Proiectul evaluează **pipeline-ul** (date → ML → API → MCP → LLM), nu acuratețea regresiei.

---

## 2. Schema de intrare — `LaptopFeatures`

Definită în [`shared/schema.py`](../shared/schema.py).

| Câmp | Tip | Constrângeri | Exemplu |
|---|---|---|---|
| `brand` | `Brand` (enum) | Dell, Asus, HP, Acer, Lenovo, MSI, Apple, Razer | `"Dell"` |
| `processor` | `Processor` (enum) | Intel i5/i7/i9, AMD Ryzen 5/7/9 | `"Intel i7"` |
| `ram_gb` | `int` | 2–256 | `16` |
| `storage_gb` | `int` | 64–8192 | `512` |
| `screen_size_inches` | `float` | 10.0–20.0 | `15.6` |
| `graphics_card` | `GraphicsCard` (enum) | Intel UHD, AMD Radeon, GTX 1650, RTX 3060, RTX 3070 | `"NVIDIA RTX 3060"` |
| `operating_system` | `OperatingSystem` (enum) | Windows 10/11, Linux, macOS | `"Windows 11"` |
| `weight_kg` | `float` | 0.5–5.0 | `1.8` |
| `battery_life_hours` | `float` | 1.0–30.0 | `8.0` |
| `warranty_years` | `int` | 0–5 | `2` |
| `model_name` *(opțional)* | `str \| None` | alias `model` | `"XPS 15"` |

---

## 3. InferenceEngine — `ml/src/inference.py`

Singleton lazy, thread-safe. Se inițializează o singură dată (~1s), apoi rămâne în memorie.

```python
from ml.src.inference import get_engine

engine = get_engine()
```

### Metode disponibile

#### `engine.predict(features: LaptopFeatures) → PredictionResponse`

Returnează prețul prezis.

```json
{
  "predicted_price_usd": 1782.50,
  "currency": "USD",
  "model_version": "1.0.0",
  "lower_bound_usd": null,
  "upper_bound_usd": null
}
```

#### `engine.explain(features: LaptopFeatures) → FeatureContributionsResponse`

Returnează contribuțiile SHAP per feature față de baseline (media pe setul de antrenament).

```json
{
  "predicted_price_usd": 1782.50,
  "baseline_usd": 1780.00,
  "contributions": [
    { "feature": "ram_gb",       "value": 32,          "shap_usd":  12.30 },
    { "feature": "processor",    "value": "Intel i9",  "shap_usd":   8.10 },
    { "feature": "graphics_card","value": "NVIDIA RTX 3060", "shap_usd": -5.20 }
  ],
  "model_version": "1.0.0"
}
```

> Contribuțiile sunt sortate descrescător după `|shap_usd|`.

#### `engine.compare(a: LaptopFeatures, b: LaptopFeatures) → ComparisonResponse`

Compară două configurații. Calculează delta de preț și impactul per feature (swap one-at-a-time).

```json
{
  "price_a_usd": 1750.00,
  "price_b_usd": 1820.00,
  "delta_usd": 70.00,
  "feature_deltas": [
    { "feature": "ram_gb", "value_a": 16, "value_b": 32, "delta_usd": 70.00 }
  ],
  "model_version": "1.0.0"
}
```

#### `engine.model_card() → ModelCard`

Returnează metadatele modelului (versiune, metrici, schema features).

```json
{
  "version": "1.0.0",
  "trained_at": "2024-01-01T00:00:00",
  "estimator": "GradientBoostingRegressor",
  "n_train": 2100,
  "n_val": 450,
  "n_test": 450,
  "val_metric_mae": 210.5,
  "val_metric_rmse": 310.2,
  "val_metric_r2": 0.01,
  "feature_columns": {
    "categorical": ["brand", "processor", "graphics_card", "operating_system"],
    "numeric": ["ram_gb", "storage_gb", "screen_size_inches", "weight_kg", "battery_life_hours", "warranty_years"]
  },
  "target": "price_usd"
}
```

---

## 4. FastAPI — endpoint-uri

Serverul rulează pe `http://localhost:8000` (default).

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "model_loaded": true, "model_version": "1.0.0" }
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Dell",
    "processor": "Intel i7",
    "ram_gb": 16,
    "storage_gb": 512,
    "screen_size_inches": 15.6,
    "graphics_card": "NVIDIA RTX 3060",
    "operating_system": "Windows 11",
    "weight_kg": 1.8,
    "battery_life_hours": 8.0,
    "warranty_years": 2
  }'
```

**Răspuns:**
```json
{
  "predicted_price_usd": 1782.50,
  "currency": "USD",
  "model_version": "1.0.0"
}
```

### `POST /explain`

Același body ca `/predict`. Returnează `FeatureContributionsResponse` (vezi secțiunea 3).

### `POST /compare`

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "a": { "brand": "Dell", "processor": "Intel i7", "ram_gb": 16, "storage_gb": 512,
           "screen_size_inches": 15.6, "graphics_card": "NVIDIA RTX 3060",
           "operating_system": "Windows 11", "weight_kg": 1.8,
           "battery_life_hours": 8.0, "warranty_years": 2 },
    "b": { "brand": "Dell", "processor": "Intel i7", "ram_gb": 32, "storage_gb": 512,
           "screen_size_inches": 15.6, "graphics_card": "NVIDIA RTX 3060",
           "operating_system": "Windows 11", "weight_kg": 1.8,
           "battery_life_hours": 8.0, "warranty_years": 2 }
  }'
```

### `GET /model-card`

```bash
curl http://localhost:8000/model-card
```

---

## 5. MCP Server — tool-uri

Serverul MCP (`mcp_server/server.py`) apelează **direct** `get_engine()` — fără hop HTTP.

| Tool MCP | Metodă engine | Răspuns |
|---|---|---|
| `predict_laptop_price` | `engine.predict()` | `PredictionResponse` ca dict JSON |
| `explain_laptop_price` | `engine.explain()` | `FeatureContributionsResponse` ca dict JSON |
| `compare_laptop_configs` | `engine.compare()` | `ComparisonResponse` ca dict JSON |
| `get_model_card` | `engine.model_card()` | `ModelCard` ca dict JSON |

**Resource:** `dataset://stats` — statistici sumare din CSV (n_rows, min/max/mean preț, distribuție brand/processor/OS).

**Prompt:** `interpret_prediction(predicted_price_usd, brand)` — ghid pentru LLM să interpreteze predicția.

### Rulare MCP

```bash
# stdio (Claude Desktop / Claude Code)
python -m mcp_server.server

# HTTP streamable (OpenWebUI)
python -m mcp_server.server --http
```

---

## 6. Dependențe necesare

```
scikit-learn
joblib
shap
numpy
pandas
pydantic
fastapi
uvicorn
mcp[cli]          # FastMCP
```

Toate sunt în `requirements.txt`.

---

## 7. Variabile de mediu

| Variabilă | Default | Descriere |
|---|---|---|
| `ARTIFACTS_DIR` | `ml/artifacts` | Director cu `model.pkl` și `model_card.json` |
| `DATASET_PATH` | `data/laptop_price_dataset.csv` | CSV necesar pentru baseline SHAP |
| `API_URL` | `http://localhost:8000` | Folosit doar dacă MCP apelează FastAPI (varianta veche) |

---

## 8. Flux complet de date

```
data/laptop_price_dataset.csv
        │
        ▼ ml/src/preprocess.py  (coercion numerice, drop rânduri corupte)
        │
        ▼ ml/src/split.py       (70/15/15, seed=42 → ml/artifacts/split.json)
        │
        ▼ ml/src/train.py       (GradientBoostingRegressor → ml/artifacts/model.pkl)
        │
        ▼ ml/src/inference.py   (InferenceEngine: predict + explain + compare + model_card)
        │
        ├──▶ api/app/main.py    (FastAPI: /predict /explain /compare /model-card /health)
        │
        └──▶ mcp_server/server.py  (FastMCP tools → LLM)
```
