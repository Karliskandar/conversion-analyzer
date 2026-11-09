# AI Website Conversion Optimization Agent

**Goal:** analyze online shopping sessions and predict whether a recorded session results in a purchase. This is a marketing analytics and session-level purchase-intent project, not a clickstream checkout funnel.

## Windows / VS Code quick start (PowerShell)

1. Open this folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Run these commands one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `py` is not recognized, install Python 3.11+ or try `python -m venv .venv`.
If `download_data.py` cannot reach UCI, download `online_shoppers_intention.csv` from the official dataset page and put it in the `data` folder; then run the train step.

Official source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Citation: Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q. License: CC BY 4.0.

## Files and ownership

- `download_data.py` — downloads UCI dataset #468; shared data source
- `train.py` — **your** baseline Random Forest, evaluation and saved model
- `app.py` — **your** home page / prediction interface
- `pages/02_Dashboard.py` — Raël's historical analytics dashboard, filters, charts and CSV exports
- `pages/03_Recommendations.py` — Raël's cohort evidence and proposed marketing A/B experiments
- `ANALYTICS_HANDOFF.md` — analytics methodology, findings, integration and validation guide
- `export_dashboard_analysis.py` — generates three presentation charts and the marketing analysis report
- `data/online_shoppers_intention.csv` — original dataset (generated)
- `models/purchase_model.joblib` — fitted model (generated)
- `reports/metrics.json` — model test metrics (generated)
- `reports/confusion_matrix.png` — figure for your presentation
- `reports/feature_importance.png` — figure for your presentation
- `reports/test_examples.csv` — held-out examples for live demo

Streamlit discovers the AI Agent, Dashboard and Recommendations pages automatically. All pages read `data/online_shoppers_intention.csv`. The analytics pages work without a trained model; their recommendations are grounded in observed associations and labeled as hypotheses for future experiments. See [ANALYTICS_HANDOFF.md](ANALYTICS_HANDOFF.md) for findings and details.

Validate the analytics and export its charts with `python -m unittest test_dashboard -v` and `python export_dashboard_analysis.py`. The exported files use the `reports/dashboard_` prefix.

## What to record for slides

- Dataset: 12,330 sessions, 17 original input variables and a binary purchase outcome (`Revenue`). The prediction model uses 16 variables, excluding `PageValues`.
- Train/test split: 80/20, stratified, `random_state=42`.
- Algorithm: RandomForestClassifier with categorical encoding and balanced class weights.
- Read actual numbers from `reports/metrics.json`; **do not invent scores**.
- Always show precision, recall, F1 and ROC-AUC (not only accuracy): only about 15.5% of original sessions end in purchases.
- Data limitation: row = session summary, not a full clickstream. No measured add-to-cart or checkout-stage abandonment.
- Prediction limitation: although `PageValues` is excluded, other features summarize behavior across the visit. This is not a start-of-visit prediction.

## Git commands (run after you have tested the model)

```powershell
git init
git add .
git commit -m "Day 1 baseline purchase prediction"
git branch -M main
```

Then create an empty repository on GitHub, add the remote URL and push:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/conversion-analyzer.git
git push -u origin main
```

Do **not** commit your `.venv` folder. Generated datasets, models and reports are excluded by `.gitignore` to keep the repository light. Both partners can regenerate them from source with `download_data.py` and `train.py`.
