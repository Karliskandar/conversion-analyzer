r"""Validate the AI Conversion Agent and find useful sessions for the Streamlit demo.

Place this file beside app.py and agent.py, then run:
    .\.venv\Scripts\python.exe test_agent_validation.py

No training, API key, or external network access is required.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from agent import ConversionAgent, format_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "online_shoppers_intention.csv"
MODEL = ROOT / "models" / "purchase_model.joblib"
EXAMPLES = ROOT / "reports" / "test_examples.csv"


def main():
    for p in (DATA, MODEL, EXAMPLES):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p.relative_to(ROOT)}. Run train.py first.")
    history = pd.read_csv(DATA)
    examples = pd.read_csv(EXAMPLES)
    assert "ActualRevenue" in examples.columns, "Missing ActualRevenue in test examples"
    assert len(examples) > 0, "Test examples file is empty"
    model = joblib.load(MODEL)
    agent = ConversionAgent(model, history)
    X = examples.drop(columns=["ActualRevenue"])
    probabilities = model.predict_proba(X)[:, 1]
    low = int(np.argmin(probabilities))
    high = int(np.argmax(probabilities))
    picks = {"Lowest probability": low, "Highest probability": high}
    medium_indices = np.where((probabilities >= 0.35) & (probabilities < 0.70))[0]
    if len(medium_indices):
        mid = int(medium_indices[np.argmin(np.abs(probabilities[medium_indices] - 0.50))])
        picks["Middle probability"] = mid

    # Identify contrasting marketing recommendations using precisely the same
    # precedence as the agent's experiment tool (without running it 2,466 times).
    bounce_cutoff = float(history["BounceRates"].quantile(0.75))
    product_cutoff = float(history["ProductRelated"].quantile(0.25))
    exit_cutoff = float(history["ExitRates"].quantile(0.75))
    experiment_labels = np.where(
        X["BounceRates"].to_numpy() > bounce_cutoff,
        "Landing-page message relevance A/B test",
        np.where(
            X["ProductRelated"].to_numpy() < product_cutoff,
            "Product discovery A/B test",
            np.where(
                X["ExitRates"].to_numpy() > exit_cutoff,
                "Clearer call-to-action A/B test",
                np.where(probabilities >= 0.70,
                         "High-intent product information A/B test",
                         "Value proposition clarity A/B test")
            )
        )
    )
    for label in ("Landing-page message relevance A/B test",
                  "Product discovery A/B test",
                  "Clearer call-to-action A/B test",
                  "High-intent product information A/B test",
                  "Value proposition clarity A/B test"):
        candidates = np.flatnonzero(experiment_labels == label)
        if len(candidates):
            picks[label] = int(candidates[0])

    print("\n=== MODEL / DATA CHECK ===")
    print(f"Held-out test sessions: {len(examples)}")
    for label, count in (
        ("Low (<35%)", int((probabilities < 0.35).sum())),
        ("Medium (35%-70%)", int(((probabilities >= .35) & (probabilities < .70)).sum())),
        ("High (>=70%)", int((probabilities >= .70).sum())),
    ):
        print(f"  {label}: {count}")

    print("\n=== RECOMMENDED STREAMLIT TEST SESSIONS ===")
    print("Session numbers below are 1-based; enter them in the app.\n")
    used = set()
    expected_tools = ["predict_purchase", "choose_strategy", "benchmark_visitors",
                      "inspect_behavior", "propose_experiments", "verify_recommendation"]
    for description, index in picks.items():
        if index in used:
            continue
        used.add(index)
        sample = X.iloc[[index]].copy()
        result = agent.run(sample)
        probability = float(probabilities[index])
        assert abs(result["purchase_probability"] - probability) < 1e-9
        assert 0 <= probability <= 1
        assert [x["tool"] for x in result["trace"]] == expected_tools
        assert len(result["checks"]) == 3
        assert result["experiment"]["status"].startswith("Proposed")
        assert result["experiment"]["variant_a"] and result["experiment"]["variant_b"]
        actual_text = str(examples.iloc[index]["ActualRevenue"]).strip().lower()
        actual = actual_text in ("1", "true")
        report = format_report(result, index + 1, actual)
        assert "Predicted purchase probability" in report
        assert "Recommended experiment" in report
        assert "Limitations" in report
        print(f"{description}:")
        print(f"  Session number: {index + 1}")
        print(f"  Probability: {probability:.1%} | Actual: {'Purchase' if actual else 'No purchase'}")
        print(f"  Strategy: {result['strategy']}")
        print(f"  Experiment: {result['experiment']['name']}")
        print(f"  Engagement signals: {len(result['signals'])}")
        print("  RESULT: PASS\n")

    if len(set(experiment_labels)) < 2:
        print("WARNING: Only one type of experiment was found on the test set.")
    else:
        print(f"Found {len(set(experiment_labels))} distinct experiment types across test sessions.")
    print("ALL SELECTED AGENT CHECKS PASSED.")
    print("Next: enter at least one low- and one high-probability session number in Streamlit;")
    print("open Agent decisions + Evidence & limitations; download a sample report.")


if __name__ == "__main__":
    main()
