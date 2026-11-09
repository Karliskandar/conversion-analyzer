"""Export actual full-data findings, CSVs and presentation charts without Streamlit.

Run: python export_dashboard_analysis.py
Outputs: reports/dashboard_* (generated and already excluded by .gitignore).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from dashboard_analysis import behavior_comparisons, conversion_by, conversion_summary, engagement_summary, load_sessions
from marketing_insights import format_marketing_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"


def main():
    data = load_sessions()
    OUT.mkdir(exist_ok=True)
    stats = conversion_summary(data)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#34465B"})
    for column, suffix, title in (("VisitorType", "visitor", "Historical conversion by visitor type"),
                                   ("Month", "month", "Conversion varies across observed months")):
        table = conversion_by(data, column)
        table.to_csv(OUT / f"dashboard_conversion_by_{suffix}.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 5.8))
        positions = range(len(table))
        bars = ax.bar(positions, table.conversion_rate, color="#16877C", width=.62)
        ax.set_xticks(list(positions), table[column].str.replace("_", " ", regex=False))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, max(.34, float(table.conversion_rate.max()) + .09))
        ax.set_ylabel("Purchases / sessions within category")
        ax.set_title(title, loc="left", pad=24)
        ax.axhline(stats["conversion_rate"], color="#5479B8", linestyle="--", linewidth=1.3,
                   label=f"All sessions: {stats['conversion_rate']:.2%}")
        ax.legend(loc="upper right", frameon=False)
        for bar, row in zip(bars, table.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .006,
                    f"{row.conversion_rate:.1%}\nn={row.sessions:,}", ha="center", va="bottom", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 1.5})
        ax.grid(axis="y", alpha=.14)
        ax.set_axisbelow(True)
        fig.text(.08, .035, f"UCI dataset 468 · {len(data):,} historical sessions · Associations, not causal effects.", fontsize=9, color="#5A6C7C")
        fig.tight_layout(rect=(0, .07, 1, 1))
        fig.savefig(OUT / f"dashboard_conversion_by_{suffix}.png", dpi=180, facecolor="white")
        plt.close(fig)

    engagement = engagement_summary(data)
    engagement.to_csv(OUT / "dashboard_engagement.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
    for ax, metric, label in zip(axes, ["BounceRates", "ExitRates", "ProductRelated"],
                               ["Median bounce metric", "Median exit metric", "Median product pages"]):
        subset = engagement.loc[engagement.metric.eq(metric)]
        ax.bar(subset.outcome, subset["median"], color=["#5479B8", "#16877C"], width=.55)
        ax.set_title(label, fontsize=12, pad=16)
        max_value = max(float(subset["median"].max()), .001)
        ax.set_ylim(0, max_value * 1.3)
        if metric != "ProductRelated":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        for i, row in enumerate(subset.itertuples()):
            value = f"{row.median:.3%}" if metric != "ProductRelated" else f"{row.median:.0f}"
            ax.text(i, row.median + max_value * .04, value, ha="center")
        ax.grid(axis="y", alpha=.14)
        ax.set_axisbelow(True)
    fig.suptitle("Purchasing sessions show different engagement patterns", x=.07, ha="left", fontweight="bold", fontsize=15)
    fig.text(.07, .07, f"No purchase: n={stats['non_purchases']:,} · Purchase: n={stats['purchases']:,}. Bounce/exit are supplied page-derived metrics.", fontsize=9, color="#5A6C7C")
    fig.text(.07, .035, "These comparisons do not establish why people buy or leave. Source: UCI dataset 468.", fontsize=9, color="#5A6C7C")
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(OUT / "dashboard_engagement.png", dpi=180, facecolor="white")
    plt.close(fig)
    behavior_comparisons(data, data).to_csv(OUT / "dashboard_behavior_comparisons.csv", index=False)
    (OUT / "dashboard_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (OUT / "dashboard_marketing_analysis.txt").write_text(format_marketing_report(data, data, "All original sessions"), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Exported three charts, four CSV tables, summary and marketing report to {OUT}")


if __name__ == "__main__":
    main()
