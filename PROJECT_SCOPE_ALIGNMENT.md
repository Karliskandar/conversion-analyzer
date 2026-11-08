# AI Website Conversion Optimization Agent — Agreed Project Scope

**Scope confirmed:** November 8, 2026  
**Project type:** AI-powered conversion analysis, session-level purchase prediction and marketing experiment recommendation.

## One-sentence description

Analyze historical e-commerce browsing sessions, estimate whether a recorded session results in a purchase, investigate patterns associated with non-conversion, and propose testable marketing improvements.

## Dataset and metrics

- Source: [UCI Online Shoppers Purchasing Intention, dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **12,330 sessions**; **1,908 purchased**; **10,422 did not purchase**.
- Historical purchase conversion rate = **1,908 / 12,330 = 15.47%** (shown as **15.5%** in the app).
- **17 original dataset features** plus the binary **Revenue** outcome; the deployed prediction model intentionally uses **16** of those features (excluding `PageValues`). One record summarizes a browsing session.
- Data contains page-category counts/durations, BounceRates, ExitRates, PageValues, visitor type, month and other session attributes.
- It does **not** contain event-by-event click paths, add-to-cart, checkout start, payment attempts or checkout completion stages.

## What the system DOES

1. Display session-level conversion KPIs and associations.
2. Use the trained Random Forest on recorded session features to estimate purchase likelihood.
3. Run a transparent Python controller that calls prediction, benchmarking, behavioral inspection, experiment selection and verification tools.
4. Suggest **unexecuted** A/B experiments, with a hypothesis, control, treatment and success metric.
5. Explain why the suggestion was made and export a report.

## What the system DOES NOT claim

- No true step-by-step shopping funnel or cart-abandonment calculation from this dataset.
- No precise identification of the checkout step where people leave.
- No proof that a page element caused abandonment or that a proposed change improves conversion.
- No live website connection, marketing automation or actual A/B test execution.
- No start-of-visit purchase prediction: although **PageValues is excluded from the deployed model**, other inputs summarize behavior throughout a visit and may only become available after browsing.
- The present tool-routing policy is **rule-based**, not LLM-selected. If the professor requires an LLM agent, that is a separate enhancement.

## Dashboard guidance — colleague workstream

**Required charts / tables (use the full dataset for descriptive historical rates):**

1. Overall conversion KPI: sessions, purchases, purchase rate.
2. Purchase rate by **VisitorType** (label as rate among sessions in each category).
3. Purchase rate by **Month** (show sample count alongside rate).
4. Engagement comparison: e.g. `ProductRelated`, `BounceRates` or `ExitRates` by purchase outcome.
5. Two or three evidence-based marketing hypotheses; keep them distinct from proven causal explanations.

**Optional:** Weekday/weekend, traffic source, PageValues distribution (add note about feature timing and interpretability).

**Do not fabricate:** add-to-cart rates, checkout-start counts, cart abandonment percentages, stepwise funnel transitions, real ROI or measured A/B uplift.

## Shared application contract

- Repository structure: `app.py`, `agent.py`, `train.py`, `pages/01_AI_Agent.py`.
- New dashboard should go in `pages/02_Dashboard.py`.
- Additional marketing page, if needed: `pages/03_Recommendations.py`.
- Read the canonical local dataset at `data/online_shoppers_intention.csv`.
- The AI model is saved to `models/purchase_model.joblib`; held-out demo sessions are in `reports/test_examples.csv`.
- Keep pages independent and avoid editing `train.py`, `agent.py`, `app.py` or the model unless agreed.
- Model preparation: stratified 80/20 train/test split; `random_state=42`; categorical encoding in a scikit-learn pipeline. **Do not run exploratory analysis only on the test split and label it the full population.**
- The dashboard should make no claim that its historical conversion rate is a future or causal estimate.

## Presentation wording

**Use:** “AI Website Conversion Optimization Agent”; “historical session-level conversion analysis”; “predict purchase likelihood from recorded session information”; “detect patterns associated with purchases”; “propose A/B experiments.”

**Avoid:** “reconstruct the complete customer funnel”; “detect exact cart abandonment”; “discover why visitors leave”; “automatically increase conversion by X%”; “LLM agent” (unless one is actually added).

### Suggested slide footnote

“Dataset limitation: session summaries, not checkout event logs. Recommendations are hypotheses for future A/B tests; no causal uplift has been measured.”

## Validation already completed

- Original 17-feature Random Forest held-out test: accuracy **88.93%**, precision **62.65%**, recall **70.68%**, F1 **66.42%**, ROC-AUC **0.9220**.
- The 16-feature ablation (excluding `PageValues`) previously gave accuracy **81.27%**, precision **38.57%**, recall **35.34%**, F1 **36.89%**, ROC-AUC **0.7613**. Confirm these values after retraining before claiming them as deployed-model metrics.
- The earlier agent tests on sessions **643**, **120** and **4** are valid for the **17-feature model only**. After the model swap, rerun `test_agent_validation.py` and select new examples.
- All agent reports must include limitations; no campaigns were launched.

## Remaining external clarification

The professor's rubric should determine whether a rule-based **ML-powered, tool-using agent** is acceptable or whether an **LLM-controlled agent** is mandatory. Do not describe the current system as LLM-driven until that feature exists.
