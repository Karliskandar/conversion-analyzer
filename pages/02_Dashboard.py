"""Raël's historical conversion dashboard. Run through `streamlit run app.py`."""
import sys
from pathlib import Path

import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_analysis import conversion_by, conversion_summary, engagement_summary
from dashboard_ui import BLUE, TEAL, cohort_filters, conversion_chart, read_history, show_conversion_table, show_kpis, source_footer
from marketing_insights import build_recommendations

st.set_page_config(page_title="Conversion dashboard", page_icon="📊", layout="wide")
st.caption("AI WEBSITE CONVERSION OPTIMIZATION AGENT")
st.title("Conversion dashboard")
st.write("Explore who purchased, when sessions converted, and how browsing behavior differs.")

history = read_history()
data, cohort_label = cohort_filters(history, "dashboard")
show_kpis(data, history)

patterns, engagement, methodology = st.tabs(["Conversion patterns", "Engagement signals", "Data & methodology"])
with patterns:
    a, b = st.columns([1, 1.45])
    for container, column, title in ((a, "VisitorType", "Conversion by visitor type"), (b, "Month", "Conversion by month")):
        with container:
            st.subheader(title)
            table = conversion_by(data, column)
            st.altair_chart(conversion_chart(table, column, conversion_summary(data)["conversion_rate"]), use_container_width=True)
            st.caption("Dashed line: selected-cohort conversion. Each bar uses its own category's session count.")
            show_conversion_table(table, column)
            st.download_button("Download " + ("visitor" if column == "VisitorType" else "monthly") + " statistics (CSV)",
                               table.to_csv(index=False), file_name=f"conversion_by_{column.lower()}.csv", mime="text/csv")
    st.caption("Only observed months are shown in calendar order. Missing months are not zero-conversion months. One year does not establish recurring seasonality.")
    st.subheader("Where to investigate next")
    plans = build_recommendations(data, history)
    if plans:
        for plan in plans:
            e = plan["evidence"]
            st.write(f"**{plan['signal']}:** {e['flagged_rate']:.2%} conversion across {e['flagged_sessions']:,} sessions, "
                     f"versus {e['other_rate']:.2%} in the other {e['other_sessions']:,} selected sessions. "
                     f"Suggested investigation: {plan['name']}.")
        st.caption("Open Recommendations in the sidebar for control/treatment variants, evidence and test design. These associations do not prove why visitors left.")
    else:
        st.info("No behavioral comparison meets the recommendation screening rules in this selection. Broaden the filters or collect more observations.")

with engagement:
    st.subheader("Browsing behavior by purchase outcome")
    st.caption("Medians describe the typical session in each outcome group. Means and sample counts are available below.")
    summary = engagement_summary(data)
    columns = st.columns(3)
    for container, metric, title, rate in zip(columns,
            ["BounceRates", "ExitRates", "ProductRelated"],
            ["Median bounce metric", "Median exit metric", "Median product pages"], [True, True, False]):
        values = summary.loc[summary["metric"].eq(metric)]
        with container:
            st.markdown(f"**{title}**")
            chart = alt.Chart(values).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("outcome:N", title=None, sort=["No purchase", "Purchase"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("median:Q", title="Median supplied metric" if rate else "Median pages", axis=alt.Axis(format=".1%" if rate else ".0f")),
                color=alt.Color("outcome:N", scale=alt.Scale(domain=["No purchase", "Purchase"], range=[BLUE, TEAL]), legend=None),
                tooltip=[alt.Tooltip("outcome:N", title="Outcome"), alt.Tooltip("sessions:Q", title="Sessions", format=","),
                         alt.Tooltip("median:Q", title="Median", format=".3%" if rate else ".1f"),
                         alt.Tooltip("mean:Q", title="Mean", format=".3%" if rate else ".2f")],
            ).properties(height=250).configure_view(stroke=None)
            st.altair_chart(chart, use_container_width=True)
    st.info("BounceRates and ExitRates are page-derived metrics supplied with each session. Their averages are not the website's overall bounce or exit percentages, and they do not identify a checkout stage.")
    table = summary.copy()
    for col in ("median", "mean"):
        table[col] = table.apply(lambda r: f"{r[col]:.3%}" if r["metric"] != "ProductRelated" else f"{r[col]:.2f}", axis=1)
    st.dataframe(table.rename(columns={"outcome": "Outcome", "metric": "Feature", "sessions": "Sessions", "median": "Median", "mean": "Mean"}),
                 hide_index=True, use_container_width=True)
    st.download_button("Download engagement statistics (CSV)", summary.to_csv(index=False), "engagement_statistics.csv", "text/csv")
    st.caption("A zero median can coexist with a nonzero mean. Associations with an observed purchase are not causal effects or arrival-time predictions.")

with methodology:
    st.markdown("""
**Unit of analysis:** one historical website session. `Revenue` is a binary purchase flag, not a monetary amount.

**Denominators:** purchases divided by sessions within each displayed category. All original rows are retained; the dashboard does not train a model, rebalance classes or use only the test split. Filters change descriptive cohorts only.

**Recommendations:** use the same full-history quartile thresholds as the agent: bounce and exit metrics strictly above their 75th percentiles, or product-page counts strictly below their 25th percentile. A proposal needs at least 30 sessions in both the flagged group and its complement, plus a lower flagged-group conversion rate. This is an exploratory screening rule, not a significance test.

**Data boundaries:** no actual cart events, checkout steps, causal diagnoses, measured uplift or ROI are available. `PageValues` is excluded from this dashboard's analysis and the deployed prediction model. Other features still summarize browsing across a visit.

**Model evaluation:** these full-data summaries are descriptive context, not held-out model metrics. The existing AI Agent page uses the separately trained Random Forest.
""")
    st.write(f"Validated source rows: **{len(history):,}**. Full-data purchasing sessions: **{int(history['RevenueBinary'].sum()):,}**.")
    duplicates = int(history.drop(columns=["RevenueBinary", "Outcome"]).duplicated().sum())
    st.write(f"Identical recorded rows: **{duplicates:,}**. Retained because identical session summaries do not establish duplicate visits; the file has no session ID to resolve them.")
    st.caption("Input validation rejects missing/invalid purchase labels, unknown months and malformed engagement values instead of silently changing denominators.")

source_footer()
