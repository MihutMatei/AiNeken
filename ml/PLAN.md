# ML Plan — Laptop Price Regressor

Target time budget: **~2 hours**. Dataset is 3000 rows × 10 features → small, fast to iterate.

---

## 1. Dataset split — three-way (train / val / test)

Frozen at the start of the session. Persist row indices so every experiment is comparable.

```
all 3000 rows
   │
   ├── train       (70%, 2100 rows) ──► fit models here
   │
   ├── validation  (15%,  450 rows) ──► model selection, hyperparameter tuning,
   │                                    quantile-band sanity, SHAP smoke tests
   │
   └── test        (15%,  450 rows) ──► TOUCHED ONCE, at the very end,
                                        for the final metrics in model_card.json
```

- **Stratified** by price quintile (`pd.qcut(y, 5, duplicates="drop")`) so all three splits cover the same price bands.
- **Seed = 42**, hard-coded.
- Done as a two-step stratified split: first peel off 15% test from the full set, then split the remaining 85% into ~82% train / ~18% val (which lands at 70/15 of the original).
- **Persist row indices** to `ml/artifacts/split.json`:
  ```json
  {"seed": 42, "train_idx": [...], "val_idx": [...], "test_idx": [...]}
  ```
  Helper module: `ml/src/split.py` (already written — see below).

**Rule:** the test set is sealed. No bake-off, no early stopping, no plotting it. Validation is where every "which is better?" decision happens. The single time you read test is to write the final metrics into `model_card.json`.

K-fold CV on the train set is no longer needed — the validation set replaces it. Faster too.

---

## 2. Training

### 2.1 Baseline → already in `ml/src/train.py`
Run it first as a sanity check. If MAE on the test split is under ~$400, stop and ship. Mean price is $1780, so that's roughly 22% — fine for a baseline.

### 2.2 Bake-off (only if baseline is weak)
Three candidates, same preprocessor (`OneHotEncoder` for categoricals, `StandardScaler` for numerics):

| Candidate | Why |
|---|---|
| `HistGradientBoostingRegressor` | Usually beats `GradientBoostingRegressor`, much faster |
| `RandomForestRegressor(n_estimators=300)` | Robust, low-tuning baseline |
| `Ridge(alpha=1.0)` | Linear floor — if it wins, you have a leakage problem |

Fit each on **train**, score on **validation** (MAE). Pick the winner. No CV needed — we have a real validation set now.

### 2.3 Light tuning on the winner
Manual loop over ~10–20 hyperparameter combinations, score each on validation, keep the best. If you want sklearn's helper, use `RandomizedSearchCV` but pass `cv=PredefinedSplit` so it uses your val set (not a fresh K-fold). Time-box to 15 min.

### 2.4 Companion quantile models (for confidence bands)
Train two extra `GradientBoostingRegressor(loss="quantile", alpha=...)` models with `alpha=0.05` and `alpha=0.95` on the same train set. These give a 90% prediction interval for free. Persist alongside the main model.

### 2.5 What gets persisted to `ml/artifacts/`
```
model.pkl              # main Pipeline — joblib.dump(pipeline)
model_q05.pkl          # lower-bound (5th percentile) quantile model
model_q95.pkl          # upper-bound (95th percentile) quantile model
explainer.pkl          # shap.TreeExplainer fitted on the trained model
split.json             # train/test row indices
model_card.json        # metrics, version, schema, feature list, training timestamp
```

`model_card.json` shape:
```json
{
  "version": "0.1.0",
  "trained_at": "2026-05-25T...",
  "estimator": "HistGradientBoostingRegressor",
  "n_train": 2100, "n_val": 450, "n_test": 450,
  "val_metric_mae": 305.1, "val_metric_rmse": 401.2, "val_metric_r2": 0.80,
  "test_metric_mae": 312.4, "test_metric_rmse": 410.8, "test_metric_r2": 0.79,
  "feature_columns": {"categorical": [...], "numeric": [...]},
  "target": "price_usd",
  "price_distribution": {"min": 501.5, "max": 2999.7, "mean": 1780.3}
}
```

---

## 3. Inference interface (how the model can be interrogated)

The point of "interrogation" is that the LLM/MCP tool should get more than a number back. Four primitives, all exposed through `ml/src/inference.py` (new module) and consumed by `api/app/main.py` and `mcp_server/server.py`:

### 3.1 `predict(features) → PredictionResponse`
Returns:
```json
{
  "predicted_price_usd": 1623.40,
  "lower_bound_usd": 1190.10,
  "upper_bound_usd": 2055.70,
  "model_version": "0.1.0",
  "currency": "USD"
}
```
Already matches `shared/schema.py::PredictionResponse` — just fill in the bounds from the quantile models.

### 3.2 `explain(features) → FeatureContributions`
Per-prediction SHAP values from `TreeExplainer`. Returns a sorted list:
```json
{
  "predicted_price_usd": 1623.40,
  "baseline_usd": 1780.33,
  "contributions": [
    {"feature": "ram_gb", "value": 16, "shap_usd": -120.5},
    {"feature": "graphics_card", "value": "NVIDIA RTX 3060", "shap_usd": +85.2},
    ...
  ]
}
```
This is the "why" — an LLM can summarize *"the price is below average mainly because of the RAM and screen size."*

### 3.3 `compare(features_a, features_b) → ComparisonResponse`
Cheap: call `predict` twice + diff the SHAP contributions per feature. Lets the LLM answer *"how much would 32GB RAM add to this configuration?"*
```json
{
  "price_a": 1623.40, "price_b": 1845.10, "delta_usd": +221.70,
  "feature_deltas": [
    {"feature": "ram_gb", "from": 16, "to": 32, "delta_usd": +180.4},
    ...
  ]
}
```

### 3.4 `model_card() → ModelCard`
Just reads `model_card.json`. Lets the LLM (or a curious user) ask *"how was this model trained, on how much data, with what error?"*

### Add to `shared/schema.py`
Two new response models for the new endpoints:
- `FeatureContribution` and `FeatureContributionsResponse`
- `FeatureDelta` and `ComparisonResponse`
- `ModelCard`

`LaptopFeatures` (the request side) **does not change**. This keeps the contract with the API/MCP people intact.

---

## 4. Execution order (~2h)

| Slot | Task | Output |
|---|---|---|
| 0:00–0:15 | EDA — open `notebooks/01_eda.ipynb`, look at target distribution, missingness, feature-target scatter. Sanity. | mental model |
| 0:15–0:25 | Run `python -m ml.src.split --data data/laptop_price_dataset.csv --out ml/artifacts/`. Verify three index sets in `split.json`. | `split.json` |
| 0:25–0:45 | Update `train.py` to read `split.json`, fit on train, report val MAE. Run it. | first `model.pkl` + val MAE |
| 0:45–1:15 | If val MAE > $400: bake-off + light tuning, all scored on val. Else: skip. | best `model.pkl` |
| 1:15–1:35 | Train q05 and q95 companions. Add `explainer = shap.TreeExplainer(model)`. Persist. | `model_q05.pkl`, `model_q95.pkl`, `explainer.pkl` |
| 1:35–1:55 | Write `ml/src/inference.py` with the four primitives. Quick smoke test from a Python REPL. | `inference.py` |
| 1:55–2:10 | Write `model_card.json`, update `ml/HANDOVER.md` with whatever changed, ping the API person. | done |

---

## 5. What this gives the API/MCP people downstream

They don't need new training knowledge — they just import:
```python
from ml.src.inference import predict, explain, compare, model_card
```
…and wrap each in a FastAPI route + MCP tool. The MCP demo prompt becomes much more impressive:

> *"Compare a Dell with 16GB vs 32GB RAM, explain why, and tell me whether the cheaper one is still worth it."*

The LLM now has `predict` + `compare` + `explain` to chain — that's the kind of dynamic-tool-use story the panel wants to see.

---

## 6. Things deliberately not in this plan
- Cross-validation on the test set (defeats the point of holding it out).
- Hyperparameter tuning beyond ~20 random configs (diminishing returns in 8h).
- Model registry, MLflow, anything beyond a `.pkl` + `.json`.
- Calibration of the prediction interval (quantile GBM intervals aren't strictly calibrated, but they're directionally useful and the panel won't grade on this).
- Categorical encoding fancier than one-hot. The cardinality is low (≤8 brands).
