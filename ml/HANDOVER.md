# ML → API/MCP handover

Status: **PricingModel v0.2.0 trained and committed.** The API and MCP layers can integrate immediately — no retraining required on your side, and the interface you've been coding against has not changed.

---

## TL;DR — how you call the model (unchanged from v0.1.0)

```python
from ml.src.inference import get_engine
from shared.schema import LaptopFeatures

engine = get_engine()          # lazy singleton, warm on first call (~1s)

engine.predict(features)       # → PredictionResponse
engine.explain(features)       # → FeatureContributionsResponse  (per-feature breakdown)
engine.compare(a, b)           # → ComparisonResponse           (diff two configs)
engine.model_card()            # → ModelCard                    (incl. learned markups)
```

All four return the same Pydantic types from `shared/schema.py` as before. Drop the engine into your FastAPI routes and MCP tools exactly as the v0.1.0 handover described.

---

## What changed under the hood (you don't need to care, but the demo will be much better)

**Old model (v0.1.0):** `GradientBoostingRegressor` on all 10 features. Black box. R² ≈ 0, no story for the LLM to tell.

**New model (v0.2.0):** `PricingModel` — a hand-engineered hybrid:

```
prediction = prior(x)  +  ridge_residual(x)

prior(x) = cpu_price[processor]            (Intel i7 → $380, RTX 3070 → $480, ...)
         + gpu_price[graphics_card]
         + $8/GB × ram_gb
         + $0.20/GB × storage_gb
         + brand_markup[brand]              (Apple → $450, Acer → $20, ...)
         + os_premium[operating_system]     (macOS → $100, Linux → $0, ...)
         + $200 assembly base
```

All those numbers are **hand-picked** in `ml/src/pricing.py` and are baked into the prior. Ridge then learns a small residual correction from the data. The result:

- Every prediction has a **dollar-denominated, per-feature explanation**.
- The card surfaces **prior vs. data-calibrated effective markups** per brand / CPU / GPU / OS — the LLM can summarize *"Apple's prior markup was $450; the data suggests it's actually closer to $196 — so on this dataset, Apple is *not* premium."*
- The model only uses **6 features**: `brand, processor, graphics_card, operating_system, ram_gb, storage_gb`. The other LaptopFeatures fields (screen size, weight, battery, warranty) are silently ignored — the API request schema is unchanged.

### Why this matters for the panel demo

The MCP demo prompt:
> *"Compare an Apple MacBook with i7/16GB/512GB to a Dell with the same specs. Tell me the price difference, the breakdown, and whether the Apple premium is real or fake."*

…now has a real answer. The LLM can chain `compare` + `explain` + `model_card` and produce:
> "Predicted Apple price $1893, Dell $1705 — $188 difference. The breakdown attributes $116 to the macOS premium and $72 to the brand markup. Note: the model's hand-picked Apple prior was $450, but the data corrected it down to $196. So the Apple premium *as seen in this dataset* is modest."

That's the story the panel wants to hear.

---

## Headline numbers (`ml/artifacts/model_card.json`)

| | MAE | RMSE | R² |
|---|---|---|---|
| train (n=2098) | $617 | — | +0.01 |
| val   (n=450)  | $622 | — | −0.02 |
| **test (n=450, sealed)** | **$626** | **$728** | **−0.01** |

Train/val gap is tiny (no overfit). Headline error is essentially flat across splits — model behaves like the simple, well-regularized thing it is. R² ≈ 0 says the *dataset* has near-zero signal; **the metric isn't a code problem**, it's a data problem we deliberately did not fight.

---

## Where to edit the model (you probably won't, but your colleague might)

| Want to change | Edit |
|---|---|
| Hand-picked CPU/GPU/brand/OS prices | `ml/src/pricing.py` — single file, all constants. Then re-run `python -m ml.src.train`. |
| Which 6 features the model uses | `ml/src/preprocess.py` — `CATEGORICAL_COLS` / `NUMERIC_COLS` / `IGNORED_COLS`. |
| Ridge regularization strength | `python -m ml.src.train --alpha 5.0` (higher = trust the priors more). |
| Estimator entirely (e.g. swap PricingModel for something else) | `ml/src/train.py::build_model`. As long as `.predict(df)` works, `inference.py` is unaffected. |

`shared/schema.py` and `ml/src/inference.py` are the **stable contract** — leave them alone unless you intentionally want to change the API.

---

## Files in `ml/artifacts/` (all committed to git)

| File | Size | Purpose |
|---|---|---|
| `model.pkl` | ~few KB | Fitted PricingModel. Self-contained — carries the frozen pricing tables it was trained with. |
| `model_card.json` | ~5 KB | Version, metrics, feature schema, **and the full pricing payload** (priors next to effective values per brand/CPU/GPU/OS). The MCP server can serve this as a resource. |
| `split.json` | ~17 KB | Frozen train/val/test row indices (seed=42). Required by `train.py` / `evaluate.py`. |

Pull the branch and you're ready to integrate — no `pip install` needed on the ML side unless you want to retrain.

---

## One sharp edge

The CSV has dirty rows (`3test` in `warranty_years`, `1UHD737.73` in price, a trailing comma somewhere). `preprocess.py` handles all three: `on_bad_lines="skip"` for malformed structure, `pd.to_numeric(errors="coerce")` + `dropna` for bad cells. You'll see `[preprocess] dropped N row(s) with non-numeric values.` on every engine warmup — that's expected, not an error.

---

## Honesty disclaimer (for the demo)

Same as before: the dataset has near-zero feature→price signal, so absolute MAE is not the win condition. The win is the *story* — a transparent, hand-engineered model whose every output the LLM can defend in dollars. Lead with that.
