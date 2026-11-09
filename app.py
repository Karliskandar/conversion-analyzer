"""Small, reliable Streamlit demo for Day 1. Teammate pages load separately."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "online_shoppers_intention.csv"
MODEL = ROOT / "models" / "purchase_model.joblib"
TEST = ROOT / "reports" / "test_examples.csv"
METRICS = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="AI Website Conversion Analyzer", page_icon="🛒", layout="wide")
st.title("🛒 AI Website Conversion Analyzer")
st.caption("University project • UCI Online Shoppers Purchasing Intention dataset")

if not DATA.exists():
    st.error("Missing dataset. Run `python download_data.py` in your terminal.")
    st.stop()

raw = pd.read_csv(DATA)
y = raw["Revenue"].astype(str).str.lower().isin(["true", "1"])
a, b, c = st.columns(3)
a.metric("Website sessions", f"{len(raw):,}")
b.metric("Sessions with purchases", f"{int(y.sum()):,}")
c.metric("Observed conversion rate", f"{100*y.mean():.1f}%")
st.info("Each row is a website session. This dataset does NOT contain a full cart-to-checkout event funnel.")

st.subheader("Purchase likelihood prediction")
if not MODEL.exists() or not TEST.exists():
    st.warning("Model not ready. Run `python train.py` to create it.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(MODEL)

@st.cache_data
def get_examples():
    return pd.read_csv(TEST)

model = get_model()
heldout = get_examples()
number = st.number_input("Choose an unseen test session", min_value=1, max_value=len(heldout), value=1, step=1)
row = heldout.iloc[[number - 1]].copy()
actual = bool(row.pop("ActualRevenue").iloc[0])

st.write("This visitor's measured session features:")
st.dataframe(row.T.rename(columns={row.index[0]: "Value"}).astype(str), width="stretch")

if st.button("Predict purchase likelihood", type="primary"):
    probability = float(model.predict_proba(row)[0, 1])
    predicted = probability >= 0.5
    col1, col2 = st.columns(2)
    col1.metric("Predicted purchase probability", f"{probability*100:.1f}%")
    col2.metric("Actual result for this held-out session", "Purchased" if actual else "Did not purchase")
    st.write(f"Prediction at a 50% threshold: **{'Purchase' if predicted else 'No purchase'}**")

st.caption(
    "The model estimates purchase probability from recorded session behavior. "
    "Feature associations cannot establish why someone leaves a website. "
    "The UCI data is historical, not a live website integration."
)
