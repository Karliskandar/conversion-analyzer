"""Cohort evidence and testable experiment briefs, aligned with the existing agent."""
from __future__ import annotations

import pandas as pd

from agent import ConversionAgent
from dashboard_analysis import MIN_COHORT, behavior_comparisons, conversion_summary

EXPERIMENTS = {
    "bounce": {
        "name": "Landing-page message relevance A/B test",
        "hypothesis": "A clearer first-screen value proposition may help visitors engage and purchase.",
        "variant_a": "Keep the current landing-page headline and call to action.",
        "variant_b": "Test a benefit-led headline and a prominent call to action.",
        "secondary_metric": "Engagement after landing, recorded prospectively.",
    },
    "product": {
        "name": "Product discovery A/B test",
        "hypothesis": "More visible product categories and related items may make suitable products easier to find.",
        "variant_a": "Keep the current product browsing layout.",
        "variant_b": "Test clearer product categories and more prominent related items.",
        "secondary_metric": "Product-page views per assigned session.",
    },
    "exit": {
        "name": "Clearer call-to-action A/B test",
        "hypothesis": "A clearer next action and more visible purchase information may improve conversion.",
        "variant_a": "Keep the current calls to action and purchase information layout.",
        "variant_b": "Test clearer calls to action with visible delivery and returns information.",
        "secondary_metric": "Next-action engagement, recorded prospectively.",
    },
}

EXPERIMENT_METHOD = (
    "Randomly assign eligible future visitors to A or B before browsing begins, keeping each visitor's "
    "assignment consistent. Compare purchases / all eligible assigned sessions in each arm. "
    "Predefine the minimum worthwhile effect, sample size, duration and guardrails before launch. "
    "Do not assign treatment using completed-session behavior or exclude non-buyers after assignment."
)
LIMITATIONS = (
    "Historical associations do not establish causes or predict treatment effects.",
    "BounceRates and ExitRates are supplied page-derived analytics metrics; their averages are not sitewide bounce or exit event counts.",
    "The data has no cart or checkout events, website content, campaign costs or measured A/B uplift.",
    "Behavioral cohorts overlap. Their session counts cannot be added as unique audiences.",
    "Rules use completed-session summaries and cannot identify an audience at arrival.",
    "These are proposed experiments. Nothing has been launched and no improvement is guaranteed.",
)


def build_recommendations(data: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    """Return proposals only when both comparison groups have >=30 sessions.

    A lower flagged-cohort rate is a screening rule, not a significance test.
    Priority describes observed non-purchasing session volume, not expected lift.
    """
    plans = []
    for row in behavior_comparisons(data, reference).to_dict("records"):
        if not row["enough_sessions"] or row["gap_pp"] >= 0:
            continue
        plan = dict(EXPERIMENTS[row["key"]])
        plan.update({"signal_key": row["key"], "signal": row["signal"], "evidence": row,
                     "success_metric": "Purchase conversion rate among all eligible assigned sessions in each A/B arm",
                     "status": "Proposed experiment only — nothing sent or launched",
                     "method": EXPERIMENT_METHOD,
                     "guardrails": "Monitor page-load performance and customer complaints; collect these in the future test.",
                     "priority_basis": "Observed non-purchasing sessions in the flagged cohort; not recoverable sales or estimated uplift."})
        # Reuse the existing agent's experiment contract without loading a model.
        plan["checks"] = ConversionAgent.verify_recommendation(plan)
        plans.append(plan)
    return sorted(plans, key=lambda p: (-p["evidence"]["flagged_non_purchases"], p["signal_key"]))


def format_marketing_report(data: pd.DataFrame, reference: pd.DataFrame, cohort_label: str) -> str:
    stats = conversion_summary(data)
    rate = f"{stats['conversion_rate']:.2%}" if stats["sessions"] else "Not available (no sessions)"
    lines = ["# AI Website Conversion Optimization Agent", "", "## Historical marketing analysis",
             f"Selection: {cohort_label}", f"Sessions: {stats['sessions']:,}",
             f"Purchases: {stats['purchases']:,}", f"Observed purchase conversion: {rate}",
             f"Behavioral thresholds: quartiles of all {len(reference):,} original sessions.", "",
             f"Screening rule: at least {MIN_COHORT} sessions in each group and a lower flagged-group conversion rate.",
             "This minimum is a descriptive safeguard, not an A/B sample-size calculation or significance test.", ""]
    plans = build_recommendations(data, reference)
    for i, plan in enumerate(plans, 1):
        e = plan["evidence"]
        lines.extend([f"## {i}. {plan['name']}",
                      f"Signal: {e['column']} strictly {e['direction']} {e['threshold']:.6g}.",
                      f"Flagged cohort: {e['flagged_purchases']:,}/{e['flagged_sessions']:,} purchases ({e['flagged_rate']:.2%}).",
                      f"Other selected sessions: {e['other_purchases']:,}/{e['other_sessions']:,} purchases ({e['other_rate']:.2%}).",
                      f"Observed difference: {e['gap_pp']:+.2f} percentage points (flagged minus other).",
                      f"Hypothesis: {plan['hypothesis']}", f"A: {plan['variant_a']}", f"B: {plan['variant_b']}",
                      f"Primary metric: {plan['success_metric']}", f"Secondary metric: {plan['secondary_metric']}",
                      f"Guardrails: {plan['guardrails']}", f"Priority: {plan['priority_basis']}", f"Status: {plan['status']}", ""])
    if not plans:
        lines.extend(["No experiment meets the evidence screening rules for this selection. Broaden the selection or collect more data.", ""])
    lines.extend(["## Future experiment design", EXPERIMENT_METHOD, "", "## Limitations"])
    lines.extend("- " + item for item in LIMITATIONS)
    lines.extend(["", "Source: Sakar & Kastro (2018), UCI dataset 468, DOI: 10.24432/C5F88Q (CC BY 4.0).", ""])
    return "\n".join(lines)
