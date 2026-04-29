# Adult — kNN classifier and comparison

UCI Adult census dataset. Predict whether a person earns `>50K` per year.

## Layout

- `data_utils.py` — shared loaders + preprocessing pipeline
- `01_eda.py` — schema, missing values, class balance, distributions
- `02_knn.py` — kNN with `GridSearchCV` (5-fold) on the train split,
  evaluated on the official `adult.test` holdout
- `03_compare.py` — kNN (best params) vs. Logistic Regression vs. Random
  Forest on the same holdout
- `results/` — text dumps, CSVs, JSON (created on first run)
- `figures/` — PNG plots (created on first run)

## Preprocessing

- Drop `fnlwgt` (sampling weight, not predictive) and `education-num`
  (redundant with `education`).
- Numeric features (`age`, `capital-gain`, `capital-loss`,
  `hours-per-week`): median imputation → `KBinsDiscretizer` with 5
  quantile bins → one-hot.
- Categorical features: most-frequent imputation → one-hot, with
  `handle_unknown="ignore"` so test-only categories don't crash.
- The same `ColumnTransformer` is used by all three classifiers so the
  comparison is on identical features.

## Run order

```bash
cd assignment
python3 01_eda.py     | tee results/01_eda.txt
python3 02_knn.py     | tee results/02_knn.txt
python3 03_compare.py | tee results/03_compare.txt
```

`02_knn.py` writes `results/knn_holdout.json`, which `03_compare.py`
picks up so the kNN row in the comparison table uses the CV-selected
hyperparameters.
