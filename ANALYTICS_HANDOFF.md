# Analytics dashboard and marketing experiments

Contribution for Raël's workstream, prepared on November 9, 2026.
Based on `main` commit `3abd04456b32cf6518c6a94fe524adc3c5e18d24`.
Local branch: `rael/analytics-dashboard`.

## What is included

| File | Purpose |
| --- | --- |
| `pages/02_Dashboard.py` | Conversion KPIs, visitor/month charts, three engagement comparisons, filters and CSV exports |
| `pages/03_Recommendations.py` | Three evidence-screened A/B experiment proposals, control/treatment variants and downloadable report |
| `dashboard_analysis.py` | Input validation, category denominators, filters and full-history behavioral thresholds |
| `dashboard_ui.py` | Shared data cache, filters, charts and display helpers |
| `marketing_insights.py` | Cohort evidence, experiment briefs and reuse of the agent's proposal verifier |
| `export_dashboard_analysis.py` | Three slide-ready PNGs, four CSVs, summary JSON and a text marketing report |
| `test_dashboard.py` | Data-calculation, methodology, boundary and Streamlit interaction checks |

No changes to `agent.py`, `train.py`, `app.py`, the prediction page or the shared requirements.
The existing README is updated for the integrated pages; `.gitignore` adds one entry for the generated marketing report.
Streamlit discovers both new pages automatically. Analytics requires the CSV, but not a trained model.
Altair is supplied by the existing Streamlit dependency. No API key is needed.

## Run on Windows

From the existing repository folder, after copying the added files into their matching paths:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Only if the canonical CSV is not already on your laptop:
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Choose **Dashboard** or **Recommendations** in the sidebar.
The existing trained model can be kept. Do not retrain it just to use the dashboard.

```powershell
.\.venv\Scripts\python.exe -m unittest test_dashboard -v
.\.venv\Scripts\python.exe test_agent_validation.py
.\.venv\Scripts\python.exe export_dashboard_analysis.py
```

The agent validation command needs the existing model and held-out examples.
The analytics tests need the canonical dataset. They exercise both new pages through the existing app's navigation.

## Findings from the actual full dataset

| Segment | Sessions | Purchases | Purchase conversion |
| --- | ---: | ---: | ---: |
| All sessions | 12,330 | 1,908 | 15.47% |
| New visitors | 1,694 | 422 | 24.91% |
| Returning visitors | 10,551 | 1,470 | 13.93% |
| Other visitor type | 85 | 16 | 18.82% |
| November | 2,998 | 760 | 25.35% |
| February | 184 | 3 | 1.63% |

New visitors have a higher observed conversion rate, while returning visitors account for most purchases because they represent most sessions. These rates alone do not establish which acquisition strategy is more profitable.

November has the highest observed monthly conversion. The dataset cannot establish whether a holiday, campaign or particular page design caused this, or whether it repeats in other years. January and April have no recorded sessions and must not be shown as zero-conversion months.

Purchasing sessions have a median of **29 product pages**, compared with **16** among non-purchasing sessions. Median supplied bounce metrics are **0.000% vs 0.426%**, and exit metrics **1.600% vs 2.857%**, respectively. These are summaries of supplied page-derived metrics, not independently measured website-wide rates.

## Experiments and evidence

All thresholds are calculated from the full original dataset, matching the agent's strict comparisons. Filters change the compared cohorts, not the thresholds.

| Proposed experiment | Flagged historical group | Purchases / sessions | Conversion | Complement conversion |
| --- | --- | ---: | ---: | ---: |
| Landing-page message relevance | BounceRates > full-data 75th percentile (~0.016813) | 144 / 3,083 | 4.67% | 19.08% |
| Clearer call to action | ExitRates > 0.05 | 80 / 2,819 | 2.84% | 19.22% |
| Product discovery | ProductRelated < 7 | 126 / 2,765 | 4.56% | 18.63% |

Each proposal includes a hypothesis, control, treatment, primary/secondary metrics and test-design guidance. Proposed experiments pass `ConversionAgent.verify_recommendation()` without a model call. No model probability is fabricated for a cohort.

- Proposals require at least 30 sessions in each compared group and a lower flagged-group conversion rate. This is exploratory screening, not statistical significance or an A/B power calculation.
- Priority is observed non-purchasing volume, not recoverable sales, monetary ROI or expected treatment effect.
- Groups overlap. Their counts cannot be added as unique visitors.
- Future A/B assignment must occur before the behavior/outcome is observed; use all eligible assigned sessions in the conversion denominator.
- No actual cart or checkout stages, website page contents, causal explanations or measured A/B uplift are available.
- All original rows are retained. Identical summaries are not automatically deleted without session IDs proving duplication.

## Presentation use

Run the export script to create:

- `reports/dashboard_conversion_by_visitor.png`
- `reports/dashboard_conversion_by_month.png`
- `reports/dashboard_engagement.png`
- `reports/dashboard_marketing_analysis.txt`
- Matching CSVs and `reports/dashboard_summary.json`.

The three chart files are ready to place into slides 3/4/7. Slides 9/10 can use the limitations above and the business value: prioritize evidence-grounded experiments and evaluate them with proper future A/B tests. The final slide deck should use the final integrated application.

## Validation performed

- All 10 checks in `python -m unittest test_dashboard -v` passed, including filter combinations, empty states, resetting, invalid inputs, rate denominators and parity with agent thresholds.
- The unchanged `test_agent_validation.py` passed all selected sessions and found five distinct experiment types with the locally regenerated model.
- Both new pages and the engagement tab were opened in a real Chromium browser through Streamlit; five charts rendered, with no browser page errors or Streamlit exceptions.
- Browser downloads of the visitor CSV and marketing report were verified against the actual counts. All 13 recommendation-page metrics rendered.
- Three exported PNG charts were visually reviewed. Source data was downloaded from UCI via the repository's `download_data.py`; 12,330 sessions and 1,908 purchases were verified.

## Reproducibility note for the model slides

The teammate reported ROC-AUC **0.7613**. Regenerating the unchanged training script here produced **0.7650** (accuracy 0.8147, precision 0.3938, recall 0.3639, F1 0.3782). The requirements allow broad dependency versions; this run used Python 3.12, pandas 2.2.3 and scikit-learn 1.8.0. The cause of the small difference has not been isolated. Use `reports/metrics.json` from the exact model used in the final demo, and record/pin that environment before presenting. This contribution does not replace the teammate's saved model.

## Remaining course clarification

The professor still needs to confirm whether a transparent ML-powered, tool-using controller satisfies the assignment or an LLM-controlled agent is required. These additions make no LLM claim.

Source: Sakar, C. & Kastro, Y. (2018), Online Shoppers Purchasing Intention, UCI dataset 468, DOI: 10.24432/C5F88Q, CC BY 4.0.
