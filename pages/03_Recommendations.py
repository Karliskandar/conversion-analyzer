"""Cohort-level marketing hypotheses that complement the single-session AI agent."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_analysis import behavior_comparisons
from dashboard_ui import cohort_filters, read_history, show_kpis, source_footer
from marketing_insights import EXPERIMENT_METHOD, LIMITATIONS, build_recommendations, format_marketing_report

st.set_page_config(page_title="Marketing recommendations", page_icon="💡", layout="wide")
st.caption("AI WEBSITE CONVERSION OPTIMIZATION AGENT")
st.title("Marketing experiments to test")
st.write("Turn observed session patterns into specific, measurable A/B test proposals.")

history = read_history()
data, cohort_label = cohort_filters(history, "recommendations")
show_kpis(data, history)
plans = build_recommendations(data, history)
st.info("Proposed experiments only. No campaigns have been launched and no conversion uplift has been measured.")
st.caption("Prioritized by observed non-purchasing session count in each flagged cohort, not predicted uplift. Cohorts overlap; do not add their counts.")

if not plans:
    st.warning("No proposal meets the screening rules: both comparison groups need at least 30 sessions and the flagged group must have a lower conversion rate. Broaden your selection or collect more data.")

for i, plan in enumerate(plans, 1):
    with st.container(border=True):
        e = plan["evidence"]
        st.subheader(f"{i}. {plan['name']}")
        st.caption(f"Signal: {e['column']} strictly {e['direction']} {e['threshold']:.6g}; threshold fixed from all {len(history):,} sessions.")
        a, b, c = st.columns(3)
        a.metric("Flagged cohort conversion", f"{e['flagged_rate']:.2%}")
        b.metric("Other selected sessions", f"{e['other_rate']:.2%}")
        c.metric("Observed gap", f"{e['gap_pp']:+.2f} pp")
        st.write(f"Evidence: **{e['flagged_purchases']:,} / {e['flagged_sessions']:,}** flagged sessions purchased, "
                 f"versus **{e['other_purchases']:,} / {e['other_sessions']:,}** other selected sessions. "
                 f"There were **{e['flagged_non_purchases']:,}** non-purchasing sessions in the flagged group.")
        st.write("**Hypothesis:** " + plan["hypothesis"])
        left, right = st.columns(2)
        left.markdown("**A · Control**\n\n" + plan["variant_a"])
        right.markdown("**B · Treatment**\n\n" + plan["variant_b"])
        st.write("**Primary success metric:** " + plan["success_metric"])
        with st.expander("Test design and verification"):
            st.write(plan["method"])
            st.write("**Secondary metric:** " + plan["secondary_metric"])
            st.write("**Guardrails:** " + plan["guardrails"])
            st.write("Verified with the existing agent's proposal checks: " + "; ".join(plan["checks"]) + ".")
        st.caption("An observed rate gap is not the improvement this experiment will produce. The website content and exact exit pages are not present in this dataset.")

with st.expander("All behavioral comparisons and screening rules"):
    comparisons = behavior_comparisons(data, history)
    st.dataframe(comparisons, hide_index=True, use_container_width=True)
    st.caption("Strict >75th / <25th percentile boundaries match the AI agent. At least 30 sessions per group is a descriptive safeguard, not statistical significance or adequate A/B test power.")
    st.download_button("Download behavioral comparisons (CSV)", comparisons.to_csv(index=False), "behavioral_comparisons.csv", "text/csv")

with st.expander("How this complements the AI agent"):
    st.write("The AI Agent page predicts one recorded session and proposes an experiment. This page uses the full labeled history (or your selected cohort) to compare groups and identify experiments worth investigating. It uses the same three behavioral thresholds and reuses the agent's proposal verifier, without retraining or loading the prediction model.")
    st.write(EXPERIMENT_METHOD)
    for item in LIMITATIONS:
        st.write("- " + item)

st.download_button("Download marketing experiment report", format_marketing_report(data, history, cohort_label),
                   "marketing_experiments.md", "text/markdown", type="primary")
source_footer()
