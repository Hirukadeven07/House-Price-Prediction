# Boston House Price Prediction

Predicts median house prices (in $1000s) on the Boston housing dataset and compares two regression models: **XGBoost** and **Random Forest**.

## Dataset

- 506 rows, 13 features, target column `PRICE`
- Split: 80% train / 20% test , `random_state=2`

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python src/train.py
python src/evaluate.py
```

`train.py` saves the models to `models/` and metrics to `reports/metrics.json`. `evaluate.py` prints the test-set comparison and saves plots to `reports/`.

Load a saved model:

```python
import joblib

model = joblib.load("models/rf_model.joblib")
```

## Results

Test set (102 held-out rows):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| XGBoost | 0.877 | 2.249 | 3.206 |
| Random Forest | 0.884 | 2.170 | 3.115 |

5-fold cross-validation (shuffled), R²:

| Model | Mean | Std |
|---|---|---|
| XGBoost | 0.862 | 0.050 |
| Random Forest | 0.868 | 0.049 |

## Findings

- **Random Forest is slightly better** on every metric, but the gap is small and within the variation between CV folds, so the two models are effectively tied.
- **Overfitting:** XGBoost with default settings reaches a training R² of about 1.0 but 0.877 on the test set. Always judge models on held-out data.
- **Cross-validation must be shuffled.** The dataset is ordered by area, so unshuffled 5-fold CV gave misleadingly low scores (0.547 and 0.619). Shuffling brought them in line with the test results.
- **Key features:** `LSTAT` (percentage of lower-status population) and `RM` (average rooms per dwelling) are the most important in both models, followed by `DIS`.
- **Typical error:** about $2,200 (MAE) on a price scale in $1000s.

## Limitations

- Small dataset (506 rows), so results vary with the split.
- `PRICE` is capped at 50, so the most expensive houses are clipped.
- Models use default or near-default hyperparameters; no tuning was done.
- The `B` feature is derived from a racial demographic formula and is considered problematic; it is kept here only for comparability with the original dataset.
