"""Train a reproducible purchase-prediction model excluding PageValues.

Run: python train.py
Input: data/online_shoppers_intention.csv
Output: models/purchase_model.joblib and reports/*
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "online_shoppers_intention.csv"
MODEL = ROOT / "models" / "purchase_model.joblib"
REPORTS = ROOT / "reports"
CATEGORIES = [
    "Month", "VisitorType", "Weekend", "OperatingSystems",
    "Browser", "Region", "TrafficType",
]


def to_binary_target(series: pd.Series) -> pd.Series:
    # Works for both a boolean column and a CSV containing True/False or 1/0.
    result = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    if result.isna().any():
        raise ValueError("Unknown label(s) in Revenue column.")
    return result.astype(int)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Dataset missing. Run: python download_data.py")
    df = pd.read_csv(DATA)
    if "Revenue" not in df.columns:
        raise ValueError("Dataset needs a Revenue column.")
    X = df.drop(columns=["Revenue", "PageValues"])
    y = to_binary_target(df["Revenue"])
    unknown = sorted(set(CATEGORIES) - set(X.columns))
    if unknown:
        raise ValueError(f"Missing columns: {unknown}")
    numbers = [c for c in X.columns if c not in CATEGORIES]

    # Label distribution is imbalanced; preserve it in the held-out test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    preprocessor = ColumnTransformer([
        ("numbers", SimpleImputer(strategy="median"), numbers),
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORIES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    scores = {
        "total_sessions": int(len(df)),
        "train_sessions": int(len(X_train)),
        "test_sessions": int(len(X_test)),
        "conversion_rate_dataset": round(float(y.mean()), 4),
        "majority_class_baseline_accuracy": round(float(max(y_test.mean(), 1-y_test.mean())), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    MODEL.parent.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    (REPORTS / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    # Use only held-out examples in the live demo, so they weren't trained on.
    examples = X_test.reset_index(drop=True).copy()
    examples["ActualRevenue"] = y_test.reset_index(drop=True)
    examples.to_csv(REPORTS / "test_examples.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No purchase", "Purchase"], cmap="Blues"
    )
    plt.title("Random Forest – held-out test set")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=180)
    plt.close()

    # Aggregate transformed one-hot column importances back to original features.
    transform = model.named_steps["preprocessor"]
    names = transform.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    ranked = pd.Series(importances, index=names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ranked.sort_values().plot.barh(ax=ax)
    ax.set_title("Top 15 model feature importances")
    ax.set_xlabel("Random Forest importance (association, not causation)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=180)
    plt.close(fig)

    print("\nTraining complete. Held-out TEST results:")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {key:12}: {scores[key]:.4f}")
    print(f"\nSaved model: {MODEL}")
    print(f"Saved outputs: {REPORTS}")
    print("Note: PageValues is EXCLUDED from training and prediction.")
    print("Other features summarize behavior across a session, so this")
    print("is NOT a beginning-of-session or causal prediction system.")


if __name__ == "__main__":
    main()
