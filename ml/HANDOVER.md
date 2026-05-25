# Handover — Part 1: Machine Learning

You own training the laptop-price regressor. ~2 hours.

## Your single deliverable
Drop two files into `ml/artifacts/`:
- `model.pkl` — a fitted scikit-learn `Pipeline` that takes a pandas DataFrame and returns price in USD.
- `model_card.json` — `{"version": "...", "metric_mae": ..., "metric_rmse": ..., "metric_r2": ..., "trained_at": "..."}`.

Once these exist, the API picks them up automatically. **Do not touch anything outside `ml/` and `data/`.**

## Setup (5 min)
```bash
cd <repo root>
pip install -r requirements.txt
python -c "from ml.src.preprocess import load_dataset; X,y = load_dataset('data/laptop_price_dataset.csv'); print(X.shape, y.shape)"
# expect: (3000, 10) (3000,)
```

## The contract — DO NOT BREAK
- Input columns to your pipeline (already produced by `ml/src/preprocess.py`):
  - **Categorical:** `brand`, `processor`, `graphics_card`, `operating_system`
  - **Numeric:** `ram_gb`, `storage_gb`, `screen_size_inches`, `weight_kg`, `battery_life_hours`, `warranty_years`
- Target: `price_usd` (float).
- Persist with `joblib.dump(pipeline, "ml/artifacts/model.pkl")` so the API can `pipe.predict(df)`.
- **Do not rename columns. Do not drop columns.** If you need to engineer features, do it inside the `Pipeline` (as a transformer step) so the API doesn't need to know.

## Baseline (already written for you)
```bash
python -m ml.src.train --data data/laptop_price_dataset.csv --out ml/artifacts/
```
This trains a `GradientBoostingRegressor` and writes both files. **Run it first** — if MAE looks acceptable, you're done.

## If MAE is too high (> ~$500)
Mean price is $1780, so MAE under $500 is roughly the bar. Things to try in `ml/src/train.py`:
1. Swap to `HistGradientBoostingRegressor` (faster, usually better).
2. Add `RandomForestRegressor` and `Ridge`, pick the winner.
3. Try `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`.
4. Quick EDA in `ml/notebooks/01_eda.ipynb` to spot leakage / weird distributions.

**Don't over-engineer.** The brief grades the pipeline, not the leaderboard. Once MAE is reasonable, ship.

## Validate before handoff
```bash
python -m ml.src.evaluate --model ml/artifacts/model.pkl --data data/laptop_price_dataset.csv
# Check ml/artifacts/model_card.json is present and metrics aren't NaN.
```

## Done when
- [ ] `ml/artifacts/model.pkl` exists
- [ ] `ml/artifacts/model_card.json` exists with non-NaN metrics
- [ ] You pinged Matei in the team chat so the API can be restarted to pick up the new model

## If you're blocked
- Pipeline errors → paste the full traceback into the AI, plus `shared/schema.py` and `ml/src/preprocess.py`.
- "Module not found" → make sure you run from the repo root, not from inside `ml/`.
- Anything ambiguous → ask Matei, don't guess and break the contract.
