# Part 1 — Machine Learning

**Owner:** _assign one teammate_
**Goal:** train a regression model that predicts `Price ($)` from laptop specs and persist it as `artifacts/model.pkl`.

## Deliverables
1. `artifacts/model.pkl` — a fitted scikit-learn `Pipeline` (preprocessor + estimator) that takes a `pandas.DataFrame` of `LaptopFeatures` (snake_case columns) and returns price in USD.
2. `artifacts/model_card.json` — `{ "version": "0.1.0", "metric_mae": ..., "metric_rmse": ..., "metric_r2": ..., "trained_at": "..." }`.
3. A short EDA notebook in `notebooks/`.

## Contract (do not break)
- Input columns: every field in `shared/schema.py::LaptopFeatures` **except** `model_name`.
- Output: a single float (USD).
- Persist as a `joblib.dump(pipeline, "artifacts/model.pkl")` — the API just calls `.predict(df)`.

## Suggested baseline
`ColumnTransformer(OneHotEncoder for categoricals, StandardScaler for numerics)` → `GradientBoostingRegressor` or `RandomForestRegressor`. Beat MAE ≈ $400 (price mean is $1780).

## Run
```bash
python -m ml.src.train --data data/laptop_price_dataset.csv --out ml/artifacts/
python -m ml.src.evaluate --model ml/artifacts/model.pkl --data data/laptop_price_dataset.csv
```
