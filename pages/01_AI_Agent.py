"""Add this to the existing `pages/` folder; do not replace app.py."""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import ConversionAgent, format_report  # noqa: E402

DATA = ROOT / "data" / "online_shoppers_intention.csv"
MODEL = ROOT / "models" / "purchase_model.joblib"
EXAMPLES = ROOT / "reports" / "test_examples.csv"

st.set_page_config(page_title="AI Conversion Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Website Conversion Optimization Agent")
st.caption("Demo: the agent selects analysis tools, evaluates a session and proposes a marketing experiment.")
st.info("This version is a **tool-using agent with transparent Python decision rules** and an ML prediction tool. It does not use an LLM or automatically launch campaigns.")

missing = [str(p.relative_to(ROOT)) for p in [DATA, MODEL, EXAMPLES] if not p.exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) + ". Run `python train.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

@st.cache_data
def load_data():
    return pd.read_csv(DATA), pd.read_csv(EXAMPLES)

history, examples = load_data()
model = load_model()

left, right = st.columns([1, 1])
with left:
    number = st.number_input(
        "Choose a held-out test session", min_value=1,
        max_value=len(examples), value=1, step=1,
    )
with right:
    st.metric("Historical conversion rate", f"{100*history['Revenue'].astype(str).str.lower().isin(['true', '1']).mean():.1f}%")

sample = examples.iloc[[number - 1]].copy()
actual = str(sample.pop("ActualRevenue").iloc[0]).strip().lower() in {"true", "1"}
with st.expander("View this session's recorded features"):
    st.dataframe(sample.T.rename(columns={sample.index[0]: "Value"}).astype(str), width="stretch")

if st.button("🚀 Run the AI agent", type="primary"):
    try:
        agent = ConversionAgent(model, history)
        st.session_state["agent_output"] = agent.run(sample)
        st.session_state["agent_demo_session"] = int(number)
        st.session_state["agent_demo_actual"] = bool(actual)
    except Exception as exc:
        st.error(f"Agent error: {exc}")
        st.stop()

result = st.session_state.get("agent_output")
if result is not None:
    if st.session_state.get("agent_demo_session") != int(number):
        st.warning("You changed the session. Click **Run the AI agent** to analyze this one.")
        st.stop()

    st.success("Agent finished: prediction → strategy → analysis → proposed A/B test → verification")
    a, b, c = st.columns(3)
    a.metric("Purchase probability", f"{result['purchase_probability']:.1%}")
    b.metric("Actual outcome", "Purchased" if st.session_state["agent_demo_actual"] else "No purchase")
    c.metric("Unusual engagement signals", len(result["signals"]))

    tab1, tab2, tab3 = st.tabs(["🎯 Recommended experiment", "🔍 Agent decisions", "📊 Evidence & limitations"])
    with tab1:
        plan = result["experiment"]
        st.subheader(plan["name"])
        st.write("**Hypothesis:**", plan["hypothesis"])
        st.write("**A (control):**", plan["variant_a"])
        st.write("**B (treatment):**", plan["variant_b"])
        st.write("**What to measure:**", plan["success_metric"])
        st.caption(plan["rationale"])
        st.warning(plan["status"])
    with tab2:
        st.write("**Goal:**", result["objective"])
        st.write("**Chosen strategy:**", result["strategy"])
        for i, step in enumerate(result["trace"], 1):
            with st.expander(f"Step {i}: {step['tool']}", expanded=(i <= 2)):
                st.write("**Why:**", step["reason"])
                st.write("**Result:**", step["observation"])
    with tab3:
        b = result["baseline"]
        st.write(f"**Historical conversion rate:** {b['all_session_conversion_rate']:.1%}")
        if b['visitor_type_conversion_rate'] is not None:
            st.write(f"**{b['visitor_type']} sessions:** {b['visitor_type_conversion_rate']:.1%} ({b['cohort_size']} sessions)")
        else:
            st.write("**Visitor-type benchmark:** not enough examples for a stable description.")
        st.write("**Signals:**")
        if result["signals"]:
            for sig in result["signals"]:
                st.write(f"- {sig['title']}: {sig['evidence']}")
        else:
            st.write("No unusually high/low indicator among the three inspected features.")
        st.write("**Limitations:**")
        for warning in result["limitations"]:
            st.write("- " + warning)

    st.download_button(
        "Download agent report",
        data=format_report(result, int(number), bool(actual)),
        file_name=f"agent_report_session_{number}.txt",
        mime="text/plain",
    )
else:
    st.write("Select a recorded session above and click **Run the AI agent** to see its tool calls and proposed marketing experiment.")
