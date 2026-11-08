"""Compare the existing Random Forest specification with and without PageValues.

Run from project root:
    python compare_models.py

This DOES NOT modify train.py, agent.py, or the live demo's saved model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train import DATA, REPORTS, CATEGORIES, to_binary_target


def build_model(columns: list[str]) -> Pipeline:
    """Match train.py's architecture and hyperparameters for a fair comparison."""
    missing_cats = sorted(set(CATEGORIES) - set(columns))
    if missing_cats:
        raise ValueError(f"Missing expected categorical fields: {missing_cats}")
    numeric_cols = [c for c in columns if c not in CATEGORIES]
    preprocessor = ColumnTransformer([
        ("numbers", SimpleImputer(strategy="median"), numeric_cols),
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORIES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(name: str, train_x: pd.DataFrame, test_x: pd.DataFrame,
             y_train: pd.Series, y_test: pd.Series) -> dict:
    model = build_model(list(train_x.columns))
    model.fit(train_x, y_train)
    predicted = model.predict(test_x)
    probabilities = model.predict_proba(test_x)[:, 1]
    return {
        "model": name,
        "features": len(train_x.columns),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA}. Put online_shoppers_intention.csv in data/."
        )
    data = pd.read_csv(DATA)
    if "Revenue" not in data.columns or "PageValues" not in data.columns:
        raise ValueError("Expected both Revenue and PageValues columns in the dataset.")
    features = data.drop(columns="Revenue")
    target = to_binary_target(data["Revenue"])

    # Exactly the same stratified split used by train.py.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=42
    )
    output = [
        evaluate("Current (with PageValues)", x_train, x_test, y_train, y_test),
        evaluate("Ablation (without PageValues)",
                 x_train.drop(columns="PageValues"),
                 x_test.drop(columns="PageValues"), y_train, y_test),
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output)
    output_path = REPORTS / "pagevalues_model_comparison.csv"
    result.to_csv(output_path, index=False)

    metrics = ["roc_auc", "f1_score", "recall"]
    ax = result.set_index("model")[metrics].T.plot.bar(figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test score (0–1)")
    ax.set_title("Random Forest model comparison: effect of removing PageValues")
    ax.legend(title="Model", loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig = ax.get_figure()
    fig.tight_layout()
    chart_path = REPORTS / "pagevalues_model_comparison.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    print("\n=== FAIR MODEL COMPARISON (SAME HELD-OUT SPLIT) ===")
    print(f"Dataset: {len(data)} sessions | Test: {len(x_test)} sessions")
    print(result.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    print(f"\nSaved table: {output_path}")
    print(f"Saved chart: {chart_path}")
    print("\nIMPORTANT: Removing PageValues does NOT make this a start-of-visit model.")
    print("Other recorded session-summary features are still used.")
    print("Your existing model and live Streamlit agent remain unchanged.")


if __name__ == "__main__":
    main()
