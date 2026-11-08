"""Small, inspectable tool-using marketing agent (no API key required).

This is NOT an LLM-driven autonomous agent: the tool-selection policy is
transparent Python. It uses the trained Random Forest as one tool and selects
additional analysis and experiment-planning tools based on its result.

Never interpret associations as causes. No campaigns are sent or launched.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


class ConversionAgent:
    """Plan -> use tools -> choose experiment -> verify output."""

    def __init__(self, model: Any, history: pd.DataFrame):
        if "Revenue" not in history:
            raise ValueError("Historical data must contain a Revenue column.")
        self.model = model
        self.history = history.copy()
        self.history["RevenueBinary"] = (
            self.history["Revenue"].astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
        )
        if self.history["RevenueBinary"].isna().any():
            raise ValueError("Historical Revenue column must be True/False or 1/0.")

    def predict_purchase(self, session: pd.DataFrame) -> float:
        """Tool 1: call the actual trained classifier on exactly one session."""
        if len(session) != 1:
            raise ValueError("Select exactly one session for the demo.")
        sample = session.drop(columns=["ActualRevenue", "Revenue", "RevenueBinary"], errors="ignore")
        expected = getattr(self.model, "feature_names_in_", None)
        if expected is not None:
            missing = [name for name in expected if name not in sample.columns]
            if missing:
                raise ValueError(f"Missing model features: {missing}")
            sample = sample[list(expected)]
        probability = float(self.model.predict_proba(sample)[0, 1])
        if not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")
        return probability

    def benchmark_visitors(self, row: pd.Series) -> dict[str, Any]:
        """Tool 2: use labeled historical sessions only as descriptive context."""
        general = float(self.history["RevenueBinary"].mean())
        visitor = str(row.get("VisitorType", "Unknown"))
        cohort = self.history[
            self.history["VisitorType"].astype(str) == visitor
        ] if "VisitorType" in self.history.columns else self.history.iloc[0:0]
        cohort_size = len(cohort)
        return {
            "all_session_conversion_rate": general,
            "visitor_type": visitor,
            "cohort_size": int(cohort_size),
            "visitor_type_conversion_rate": (
                float(cohort["RevenueBinary"].mean()) if cohort_size >= 30 else None
            ),
            "note": "Descriptive historical comparison, not a causal explanation or an individualized treatment effect.",
        }

    def inspect_behavior(self, row: pd.Series) -> list[dict[str, str]]:
        """Tool 3: flag *unusual* session statistics using historical quartiles."""
        signals: list[dict[str, str]] = []
        rules = (
            ("BounceRates", "high", "Relatively high bounce rate", "bounce"),
            ("ExitRates", "high", "Relatively high exit rate", "exit"),
            ("ProductRelated", "low", "Relatively few product pages viewed", "product"),
        )
        for col, direction, description, key in rules:
            if col not in self.history.columns or col not in row.index:
                continue
            ref = pd.to_numeric(self.history[col], errors="coerce").dropna()
            if len(ref) < 30:
                continue
            observed = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(observed):
                continue
            threshold = float(ref.quantile(.75 if direction == "high" else .25))
            is_flagged = observed > threshold if direction == "high" else observed < threshold
            if is_flagged:
                signals.append({
                    "key": key,
                    "title": description,
                    "evidence": f"{col} = {float(observed):.3f}; historical {'75th' if direction == 'high' else '25th'} percentile = {threshold:.3f}.",
                })
        return signals

    def propose_experiments(self, probability: float, signals: list[dict[str, str]]) -> dict[str, Any]:
        """Tool 4: choose an A/B test hypothesis, not a guaranteed solution."""
        keys = {x["key"] for x in signals}
        if "bounce" in keys:
            action = {
                "name": "Landing-page message relevance A/B test",
                "hypothesis": "A more relevant headline and clearer first-screen value proposition may help visitors engage.",
                "variant_a": "Keep the current landing-page headline.",
                "variant_b": "Test a clearer benefit-led headline and prominent call to action.",
                "success_metric": "Purchase conversion rate (with engagement as a secondary metric)",
                "rationale": "The session's bounce rate is unusually high. This is a signal to investigate, not proof that the headline caused the bounce.",
            }
        elif "product" in keys:
            action = {
                "name": "Product discovery A/B test",
                "hypothesis": "Improving visibility of relevant products may encourage deeper browsing.",
                "variant_a": "Keep the current product browsing layout.",
                "variant_b": "Test more prominent product categories and related items.",
                "success_metric": "Purchase conversion rate (with product-page views as a secondary metric)",
                "rationale": "The session has relatively few product page views; this does not establish why.",
            }
        elif "exit" in keys:
            action = {
                "name": "Clearer call-to-action A/B test",
                "hypothesis": "Clarifying the next step may reduce friction in the shopping experience.",
                "variant_a": "Keep the current call to action.",
                "variant_b": "Test a clearer call to action and more visible purchase information.",
                "success_metric": "Purchase conversion rate",
                "rationale": "The session's exit rate is relatively high, but the dataset does not reveal the exact exit page or checkout stage.",
            }
        elif probability >= .70:
            action = {
                "name": "High-intent product information A/B test",
                "hypothesis": "Better product information may help engaged visitors make a decision.",
                "variant_a": "Keep the existing product information layout.",
                "variant_b": "Test a concise benefits comparison and more visible return-policy details.",
                "success_metric": "Purchase conversion rate",
                "rationale": "The model ranks this session as relatively likely to purchase; the proposed test is exploratory.",
            }
        else:
            action = {
                "name": "Value proposition clarity A/B test",
                "hypothesis": "A more specific value proposition may improve purchase decisions.",
                "variant_a": "Keep the current product messaging.",
                "variant_b": "Test clearer benefits, pricing details, and a visible call to action.",
                "success_metric": "Purchase conversion rate",
                "rationale": "There is no strongly flagged behavioral signal in the selected fields; this is a general test idea.",
            }
        action["status"] = "Proposed experiment only — nothing sent or launched"
        return action

    @staticmethod
    def verify_recommendation(plan: dict[str, Any]) -> list[str]:
        """Tool 5: check the suggested action can be tested and is not launched."""
        checks = []
        if plan.get("variant_a") and plan.get("variant_b"):
            checks.append("Has control and treatment variants")
        if plan.get("success_metric"):
            checks.append("Has a measurable outcome")
        if plan.get("status", "").startswith("Proposed"):
            checks.append("No automatic campaign launch or customer targeting")
        if len(checks) != 3:
            raise ValueError("Experiment proposal failed basic verification.")
        return checks

    def run(self, session: pd.DataFrame) -> dict[str, Any]:
        """Agent chooses its tool path conditionally and records its decisions."""
        trace: list[dict[str, str]] = []

        def record(tool: str, reason: str, observation: str) -> None:
            trace.append({"tool": tool, "reason": reason, "observation": observation})

        row = session.iloc[0]
        probability = self.predict_purchase(session)
        record("predict_purchase", "Estimate purchase likelihood using the trained Random Forest.", f"Model probability: {probability:.1%}")

        if probability < .35:
            route = "low purchase-likelihood branch"
            goal = "Investigate low engagement and propose a conversion experiment."
        elif probability < .70:
            route = "medium purchase-likelihood branch"
            goal = "Investigate behavior and propose an experiment worth testing."
        else:
            route = "high purchase-likelihood branch"
            goal = "Identify a relevant test for relatively engaged sessions."
        record("choose_strategy", "Route automatically based on the model's output.", route)

        baseline = self.benchmark_visitors(row)
        cohort_str = (
            f"Visitor-type conversion: {baseline['visitor_type_conversion_rate']:.1%} over {baseline['cohort_size']} sessions"
            if baseline["visitor_type_conversion_rate"] is not None
            else "Visitor-type comparison unavailable: fewer than 30 sessions"
        )
        record("benchmark_visitors", "Compare this session's category with observed historical results.", cohort_str)

        signals = self.inspect_behavior(row)
        if probability < .70 or signals:
            record("inspect_behavior", "Examine unusual engagement patterns.", f"Found {len(signals)} flagged indicator(s).")
        else:
            # Still inspect in order to make a fully grounded choice; the trace
            # explicitly distinguishes the high-intent branch.
            record("inspect_behavior", "Check whether any obvious friction flags override the high-intent strategy.", f"Found {len(signals)} flagged indicator(s).")

        experiment = self.propose_experiments(probability, signals)
        record("propose_experiments", "Pick a test based on flagged signals and estimated likelihood.", experiment["name"])
        checks = self.verify_recommendation(experiment)
        record("verify_recommendation", "Check testability and avoid unsupported promises.", "; ".join(checks))

        return {
            "objective": "Find a defensible website conversion experiment to test",
            "strategy": route,
            "goal": goal,
            "purchase_probability": probability,
            "baseline": baseline,
            "signals": signals,
            "experiment": experiment,
            "checks": checks,
            "trace": trace,
            "limitations": [
                "The model excludes PageValues but still uses recorded browsing-session summaries; it is not an arrival-time predictor.",
                "Historical correlations and model scores do not demonstrate why a visitor did or did not buy.",
                "This dataset does not include individual checkout-step events.",
                "The agent proposes an A/B test; it does not run campaigns or prove business lift.",
                "The tool-routing strategy is rule-based; no LLM decides the tool calls in this version.",
            ],
        }


def format_report(result: dict[str, Any], session_number: int, actual: bool) -> str:
    """Plain-text report suitable for Streamlit's download button."""
    plan = result["experiment"]
    lines = [
        "AI WEBSITE CONVERSION OPTIMIZATION AGENT — DEMO REPORT",
        f"Held-out test session: {session_number}",
        f"Predicted purchase probability: {result['purchase_probability']:.1%}",
        f"Actual historical outcome: {'Purchased' if actual else 'No purchase'}",
        f"Agent strategy: {result['strategy']}",
        "", "Flagged patterns:",
    ]
    lines.extend([f"- {x['title']}: {x['evidence']}" for x in result["signals"]] or ["- None in selected fields."])
    lines.extend([
        "", "Recommended experiment (not executed):", f"- {plan['name']}",
        f"- Hypothesis: {plan['hypothesis']}",
        f"- Control: {plan['variant_a']}",
        f"- Variant: {plan['variant_b']}",
        f"- Success metric: {plan['success_metric']}",
        "", "Agent tool execution:",
    ])
    lines.extend([f"- {x['tool']}: {x['observation']}" for x in result["trace"]])
    lines.extend(["", "Limitations:"] + [f"- {x}" for x in result["limitations"]])
    return "\n".join(lines) + "\n"
