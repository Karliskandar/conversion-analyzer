"""Analytics and multipage interaction checks. Run: python -m unittest test_dashboard -v"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from agent import ConversionAgent
from dashboard_analysis import (DATA_PATH, behavior_comparisons, conversion_by, conversion_summary,
                                filter_sessions, load_sessions, prepare_sessions)
from marketing_insights import build_recommendations, format_marketing_report

ROOT = Path(__file__).resolve().parent


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = load_sessions()

    def test_original_counts_and_group_denominators(self):
        stats = conversion_summary(self.history)
        self.assertEqual((stats["sessions"], stats["purchases"], stats["non_purchases"]), (12330, 1908, 10422))
        self.assertAlmostEqual(stats["conversion_rate"], 1908 / 12330)
        for column in ("Month", "VisitorType"):
            groups = conversion_by(self.history, column)
            self.assertEqual(groups.sessions.sum(), 12330)
            self.assertEqual(groups.purchases.sum(), 1908)
            self.assertTrue((groups.conversion_rate == groups.purchases / groups.sessions).all())
        self.assertEqual(conversion_by(self.history, "Month")["Month"].tolist(),
                         ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    def test_filter_combination_and_empty_selection(self):
        selected = filter_sessions(self.history, ["Nov"], "Returning_Visitor", "Weekends")
        expected = self.history.loc[(self.history.Month == "Nov") & (self.history.VisitorType == "Returning_Visitor") & self.history.Weekend]
        pd.testing.assert_frame_equal(selected, expected)
        empty = filter_sessions(self.history, [])
        self.assertIsNone(conversion_summary(empty)["conversion_rate"])
        self.assertEqual(build_recommendations(empty, self.history), [])
        self.assertIn("no sessions", format_marketing_report(empty, self.history, "empty"))

    def test_reject_bad_labels_and_engagement_values(self):
        raw = pd.read_csv(DATA_PATH).head(5)
        for column, bad in (("Revenue", "unknown"), ("Revenue", None), ("Weekend", "maybe"),
                            ("BounceRates", 1.2), ("ExitRates", -0.1), ("ProductRelated", float("inf")),
                            ("ProductRelated", 1.5), ("Month", "Unknown")):
            with self.subTest(column=column, bad=bad):
                invalid = raw.astype({column: object}).copy()
                invalid.loc[0, column] = bad
                with self.assertRaises(ValueError):
                    prepare_sessions(invalid)

    def test_labels_and_duplicate_rows_are_preserved(self):
        raw = pd.read_csv(DATA_PATH).head(4)
        raw["Revenue"] = [" True ", "FALSE", "1", "0"]
        original = raw.copy(deep=True)
        prepared = prepare_sessions(raw)
        self.assertEqual(prepared.RevenueBinary.tolist(), [1, 0, 1, 0])
        pd.testing.assert_frame_equal(raw, original)
        twice = pd.concat([raw, raw], ignore_index=True)
        self.assertEqual(len(prepare_sessions(twice)), 8)

    def test_thresholds_and_boundaries_match_existing_agent(self):
        comparisons = behavior_comparisons(self.history, self.history)
        agent = ConversionAgent(None, self.history)
        for item in comparisons.to_dict("records"):
            row = pd.Series({"BounceRates": 0., "ExitRates": 0., "ProductRelated": 1000.})
            row[item["column"]] = item["threshold"]
            self.assertNotIn(item["key"], {s["key"] for s in agent.inspect_behavior(row)})
            row[item["column"]] += .001 if item["direction"] == "above" else -.001
            self.assertIn(item["key"], {s["key"] for s in agent.inspect_behavior(row)})
            self.assertEqual(item["flagged_sessions"] + item["other_sessions"], len(self.history))
        subset = self.history.loc[self.history.Month.eq("Nov")]
        self.assertEqual(behavior_comparisons(subset, self.history).threshold.tolist(), comparisons.threshold.tolist())

    def test_proposals_use_real_evidence_and_existing_verifier(self):
        plans = build_recommendations(self.history, self.history)
        self.assertEqual({p["signal_key"] for p in plans}, {"bounce", "product", "exit"})
        for plan in plans:
            e = plan["evidence"]
            self.assertLess(e["gap_pp"], 0)
            self.assertGreaterEqual(min(e["flagged_sessions"], e["other_sessions"]), 30)
            self.assertAlmostEqual(e["flagged_rate"], e["flagged_purchases"] / e["flagged_sessions"])
            self.assertEqual(len(ConversionAgent.verify_recommendation(plan)), 3)
        self.assertEqual(build_recommendations(self.history.head(20), self.history), [])
        report = format_marketing_report(self.history, self.history, "all sessions")
        for term in ("12,330", "1,908", "15.47%", "Limitations", "nothing sent or launched"):
            self.assertIn(term, report)

    def test_no_proposal_for_equal_conversion_rates(self):
        raw = pd.read_csv(DATA_PATH).head(4)
        raw["BounceRates"] = [.01, .01, .10, .10]
        raw["ExitRates"] = [.01, .01, .10, .10]
        raw["ProductRelated"] = [1, 1, 30, 30]
        raw["Revenue"] = [0, 1, 0, 1]
        data = prepare_sessions(pd.concat([raw] * 40, ignore_index=True))
        self.assertEqual(build_recommendations(data, self.history), [])


class PageTests(unittest.TestCase):
    def page(self, name):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        return app.switch_page("pages/" + name).run()

    def test_dashboard_filters_empty_state_reset_and_charts(self):
        app = self.page("02_Dashboard.py")
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual([x.value for x in app.metric], ["12,330", "1,908", "15.47%", "10,422"])
        # Streamlit renamed this element in newer releases; support the repo's range.
        self.assertEqual(len(app.get("arrow_vega_lite_chart")) + len(app.get("vega_lite_chart")), 5)
        app.selectbox(key="dashboard_visitor").set_value("New_Visitor").run()
        history = load_sessions()
        selected = history.loc[history.VisitorType.eq("New_Visitor")]
        self.assertEqual(app.metric[0].value, f"{len(selected):,}")
        app.multiselect(key="dashboard_months").set_value([]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("No sessions", app.info[0].value)
        app.button(key="dashboard_reset").click().run()
        self.assertEqual(app.metric[0].value, "12,330")

    def test_recommendations_and_small_cohort(self):
        app = self.page("03_Recommendations.py")
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertTrue(any("Landing-page" in x.value for x in app.subheader))
        app.selectbox(key="recommendations_visitor").set_value("Other").run()
        app.multiselect(key="recommendations_months").set_value(["Feb"]).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertFalse(any("A/B test" in x.value for x in app.subheader))

    def test_missing_and_invalid_dataset_messages(self):
        import dashboard_ui
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.csv"
            with patch.object(dashboard_ui, "DATA_PATH", path):
                app = self.page("02_Dashboard.py")
                self.assertEqual(len(app.exception), 0)
                self.assertIn("Dataset missing", app.error[0].value)
            path.write_text("Revenue,Month\nunknown,Nov\n", encoding="utf-8")
            with patch.object(dashboard_ui, "DATA_PATH", path):
                app = self.page("02_Dashboard.py")
                self.assertEqual(len(app.exception), 0)
                self.assertIn("valid session dataset", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
