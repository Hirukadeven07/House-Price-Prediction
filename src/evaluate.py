import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from train import MODELS_DIR, REPORTS_DIR, load_data, score, split_data

MODEL_FILES = {
    "XGBoost": "xgb_model.joblib",
    "Random Forest": "rf_model.joblib",
}
COLORS = {"XGBoost": "tab:orange", "Random Forest": "tab:green"}


def load_models():
    return {name: joblib.load(MODELS_DIR / file) for name, file in MODEL_FILES.items()}


def plot_actual_vs_predicted(y_test, predictions):
    fig, axes = plt.subplots(1, len(predictions), figsize=(12, 5), sharex=True, sharey=True)
    lims = [y_test.min(), y_test.max()]
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, pred, color=COLORS[name], alpha=0.7)
        ax.plot(lims, lims, "r--")
        ax.set_title(f"{name} (R2 = {score(y_test, pred)['r2']:.3f})")
        ax.set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_metrics(results):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["r2", "mae", "rmse"]):
        ax.bar(results.index, results[metric], color=[COLORS[n] for n in results.index])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(models, columns):
    importances = pd.DataFrame(
        {name: model.feature_importances_ for name, model in models.items()},
        index=columns,
    ).sort_values("Random Forest")
    ax = importances.plot(kind="barh", figsize=(8, 6), color=[COLORS[c] for c in importances.columns])
    ax.set_title("Feature Importance: XGBoost vs Random Forest")
    plt.tight_layout()
    ax.get_figure().savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(ax.get_figure())


def main():
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)

    models = load_models()
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    results = pd.DataFrame({name: score(y_test, pred) for name, pred in predictions.items()}).T

    REPORTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORTS_DIR / "test_metrics.csv")
    print(results.round(4))

    plot_actual_vs_predicted(y_test, predictions)
    plot_metrics(results)
    plot_feature_importance(models, X.columns)

    best = results["r2"].idxmax()
    print(f"\nBest model on test set: {best}")
    print(f"plots saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()