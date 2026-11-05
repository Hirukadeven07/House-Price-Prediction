import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Boston House Price Data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TARGET = "PRICE"
SEED = 2
TEST_SIZE = 0.2


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def build_models():
    return {
        "xgb_model": XGBRegressor(random_state=SEED),
        "rf_model": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    }


def score(y_true, y_pred):
    return {
        "r2": float(metrics.r2_score(y_true, y_pred)),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
    }


def cross_validate(models, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        results[name] = {
            "folds": [float(s) for s in scores],
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
    return results


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X.shape, X_train.shape, X_test.shape)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    models = build_models()
    report = {"train": {}, "test": {}}

    for name, model in models.items():
        model.fit(X_train, y_train)
        report["train"][name] = score(y_train, model.predict(X_train))
        report["test"][name] = score(y_test, model.predict(X_test))
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"saved {MODELS_DIR / (name + '.joblib')}")

    report["cross_validation"] = cross_validate(build_models(), X, y)

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    for split in ("train", "test"):
        print(f"\n{split.upper()}")
        print(pd.DataFrame(report[split]).round(4))

    print("\nCROSS-VALIDATION R2")
    for name, res in report["cross_validation"].items():
        print(f"{name}: mean={res['mean']:.4f} std={res['std']:.4f}")


if __name__ == "__main__":
    main()