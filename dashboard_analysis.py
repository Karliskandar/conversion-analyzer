"""Descriptive analytics for the canonical UCI session data; no model required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MIN_COHORT = 30
BEHAVIOR_RULES = (
    ("bounce", "BounceRates", "High bounce metric", .75, "above"),
    ("product", "ProductRelated", "Few product pages", .25, "below"),
    ("exit", "ExitRates", "High exit metric", .75, "above"),
)


def binary_labels(series: pd.Series, name: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    )
    if values.isna().any():
        raise ValueError(f"{name} must contain only True/False or 1/0, with no missing labels.")
    return values.astype(int)


def prepare_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate before computing denominators. Keep every original session row."""
    required = {"Revenue", "Month", "VisitorType", "Weekend", "BounceRates", "ExitRates", "ProductRelated"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if raw.empty:
        raise ValueError("The dataset contains no sessions.")
    data = raw.copy()
    data["RevenueBinary"] = binary_labels(data["Revenue"], "Revenue")
    data["Weekend"] = binary_labels(data["Weekend"], "Weekend").astype(bool)
    for col in ("Month", "VisitorType"):
        if data[col].isna().any() or data[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} has missing or blank categories.")
        data[col] = data[col].astype(str).str.strip()
    data["Month"] = data["Month"].replace({"Jun": "June"})
    if not data["Month"].isin(MONTH_ORDER).all():
        raise ValueError("Month contains an unrecognized category.")
    for col in ("BounceRates", "ExitRates", "ProductRelated"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if not np.isfinite(data[col]).all() or data[col].lt(0).any():
            raise ValueError(f"{col} must contain finite, non-negative numbers.")
    if data[["BounceRates", "ExitRates"]].gt(1).any().any():
        raise ValueError("BounceRates and ExitRates must be fractions between 0 and 1.")
    if data["ProductRelated"].mod(1).ne(0).any():
        raise ValueError("ProductRelated must contain whole page counts.")
    data["Outcome"] = data["RevenueBinary"].map({0: "No purchase", 1: "Purchase"})
    return data


def load_sessions(path: str | Path = DATA_PATH) -> pd.DataFrame:
    return prepare_sessions(pd.read_csv(path))


def filter_sessions(data: pd.DataFrame, months: list[str], visitor: str = "All visitors", days: str = "All days") -> pd.DataFrame:
    mask = data["Month"].isin(months)
    if visitor != "All visitors":
        mask &= data["VisitorType"].eq(visitor)
    if days == "Weekdays":
        mask &= ~data["Weekend"]
    elif days == "Weekends":
        mask &= data["Weekend"]
    return data.loc[mask].copy()


def conversion_summary(data: pd.DataFrame) -> dict:
    n = len(data)
    purchases = int(data["RevenueBinary"].sum())
    return {"sessions": n, "purchases": purchases, "non_purchases": n - purchases,
            "conversion_rate": purchases / n if n else None}


def conversion_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """The denominator is the sessions within each displayed category."""
    table = data.groupby(column, observed=True)["RevenueBinary"].agg(sessions="size", purchases="sum").reset_index()
    if column == "Month":
        table = table.sort_values("Month", key=lambda values: values.map({m: i for i, m in enumerate(MONTH_ORDER)}))
    table["conversion_rate"] = table["purchases"] / table["sessions"]
    table["sample_note"] = np.where(table["sessions"] < MIN_COHORT, "Small sample (<30)", "30+ sessions")
    return table.reset_index(drop=True)


def engagement_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ("No purchase", "Purchase"):
        cohort = data.loc[data["Outcome"].eq(outcome)]
        if cohort.empty:
            continue
        for col in ("BounceRates", "ExitRates", "ProductRelated"):
            rows.append({"outcome": outcome, "metric": col, "sessions": len(cohort),
                         "median": float(cohort[col].median()), "mean": float(cohort[col].mean())})
    return pd.DataFrame(rows, columns=["outcome", "metric", "sessions", "median", "mean"])


def behavior_comparisons(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Use the agent's strict quartile rules, fixed against the full history.

    Each flagged cohort is compared to its complement within the current filter.
    Cohorts can overlap across rules and must never be added as unique visitors.
    """
    rows = []
    for key, column, label, quantile, direction in BEHAVIOR_RULES:
        threshold = float(reference[column].quantile(quantile))
        mask = data[column].gt(threshold) if direction == "above" else data[column].lt(threshold)
        flagged, other = conversion_summary(data.loc[mask]), conversion_summary(data.loc[~mask])
        difference = None
        if flagged["sessions"] and other["sessions"]:
            difference = 100 * (flagged["conversion_rate"] - other["conversion_rate"])
        rows.append({"key": key, "column": column, "signal": label, "threshold": threshold,
                     "direction": direction, "quantile": quantile,
                     "flagged_sessions": flagged["sessions"], "flagged_purchases": flagged["purchases"],
                     "flagged_non_purchases": flagged["non_purchases"], "flagged_rate": flagged["conversion_rate"],
                     "other_sessions": other["sessions"], "other_purchases": other["purchases"],
                     "other_rate": other["conversion_rate"], "gap_pp": difference,
                     "enough_sessions": min(flagged["sessions"], other["sessions"]) >= MIN_COHORT})
    return pd.DataFrame(rows)
