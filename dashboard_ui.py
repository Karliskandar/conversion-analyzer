"""Shared, model-independent UI helpers for the two analytics pages."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard_analysis import DATA_PATH, MONTH_ORDER, conversion_summary, filter_sessions, load_sessions

TEAL = "#16877C"
BLUE = "#5479B8"
MUTED = "#98A5B5"


@st.cache_data(show_spinner=False)
def _cached_sessions(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    # File metadata participates in the cache key so replacing the CSV refreshes it.
    return load_sessions(path)


def read_history() -> pd.DataFrame:
    try:
        stat = DATA_PATH.stat()
        return _cached_sessions(str(DATA_PATH), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        st.error("Dataset missing. Run `python download_data.py` from the project folder, then reload this page.")
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not read a valid session dataset: {exc}")
    st.stop()


def cohort_filters(history: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, str]:
    months = [m for m in MONTH_ORDER if m in set(history["Month"])]
    def reset() -> None:
        st.session_state[f"{prefix}_months"] = months
        st.session_state[f"{prefix}_visitor"] = "All visitors"
        st.session_state[f"{prefix}_days"] = "All days"

    with st.sidebar:
        st.subheader("Explore a cohort")
        st.caption("Filters apply to this page. Start with all historical sessions.")
        visitor = st.selectbox("Visitor type", ["All visitors"] + sorted(history["VisitorType"].unique()),
                               format_func=lambda x: x.replace("_", " "), key=f"{prefix}_visitor")
        selected_months = st.multiselect("Months", months, default=months, key=f"{prefix}_months")
        days = st.radio("Visit days", ["All days", "Weekdays", "Weekends"], key=f"{prefix}_days")
        st.button("Reset filters", on_click=reset, key=f"{prefix}_reset")
    data = filter_sessions(history, selected_months, visitor, days)
    month_label = "all observed months" if set(selected_months) == set(months) else ", ".join(selected_months) or "no months"
    label = f"{visitor.replace('_', ' ')} · {month_label} · {days.lower()}"
    st.caption(f"Selection: {label} | {len(data):,} of {len(history):,} sessions")
    if data.empty:
        st.info("No sessions match these filters. Choose at least one month or reset the filters.")
        st.stop()
    if len(data) < 30:
        st.warning("This selection contains fewer than 30 sessions. Treat its rates as a small-sample description.")
    return data, label


def show_kpis(data: pd.DataFrame, history: pd.DataFrame) -> None:
    stats, baseline = conversion_summary(data), conversion_summary(history)
    a, b, c, d = st.columns(4)
    a.metric("Sessions", f"{stats['sessions']:,}")
    b.metric("Purchasing sessions", f"{stats['purchases']:,}")
    c.metric("Historical conversion", f"{stats['conversion_rate']:.2%}")
    d.metric("Non-purchasing sessions", f"{stats['non_purchases']:,}")
    st.caption(
        f"Purchase conversion = purchasing sessions ÷ all sessions in the selection. "
        f"Full-data reference: {baseline['conversion_rate']:.2%}."
    )


def conversion_chart(table: pd.DataFrame, column: str, baseline: float) -> alt.LayerChart:
    plot = table.copy()
    plot["label"] = plot[column].astype(str).str.replace("_", " ", regex=False)
    plot["annotation"] = plot.apply(lambda r: f"{r['conversion_rate']:.1%} · n={r['sessions']:,}", axis=1)
    order = plot["label"].tolist()
    base = alt.Chart(plot).encode(
        x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=170)),
        y=alt.Y("conversion_rate:Q", title="Purchase conversion", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, min(1.08, max(.35, float(plot["conversion_rate"].max()) + .10))])),
        tooltip=[alt.Tooltip("label:N", title="Category"), alt.Tooltip("sessions:Q", title="Sessions", format=","),
                 alt.Tooltip("purchases:Q", title="Purchases", format=","),
                 alt.Tooltip("conversion_rate:Q", title="Conversion", format=".2%"),
                 alt.Tooltip("sample_note:N", title="Sample")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color("sample_note:N", scale=alt.Scale(domain=["30+ sessions", "Small sample (<30)"], range=[TEAL, MUTED]), legend=None)
    )
    # Only annotate the few visitor categories. Month counts remain in the table and hover tooltip.
    layers = [bars]
    if column != "Month":
        layers.append(base.mark_text(dy=-20, color="#617080", fontSize=12).encode(text="annotation:N"))
    layers.insert(1, alt.Chart(pd.DataFrame({"reference": [baseline]})).mark_rule(color=BLUE, strokeDash=[5, 4]).encode(y="reference:Q"))
    return alt.layer(*layers).properties(height=290).configure_view(stroke=None)


def show_conversion_table(table: pd.DataFrame, column: str) -> None:
    display = table.rename(columns={column: "Category", "sessions": "Sessions", "purchases": "Purchases", "sample_note": "Sample"}).copy()
    display["Category"] = display["Category"].astype(str).str.replace("_", " ", regex=False)
    display["Conversion"] = display["conversion_rate"].map(lambda x: f"{x:.2%}")
    st.dataframe(display[["Category", "Sessions", "Purchases", "Conversion", "Sample"]], hide_index=True, use_container_width=True)


def source_footer() -> None:
    st.caption("Source: Sakar & Kastro (2018), UCI Online Shoppers Purchasing Intention · CC BY 4.0 · DOI: 10.24432/C5F88Q")
