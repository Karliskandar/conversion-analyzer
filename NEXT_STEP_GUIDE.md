# Next step — Model robustness check (without changing the working agent)

## Goal
Compare your current Random Forest model with a second Random Forest trained **without PageValues**. This is a **feature ablation**, not a causal experiment. It tests how dependent model performance is on that field. Removing PageValues does **not** make the remaining session-level features available when the visitor first lands on the site.

## Windows/VS Code instructions

1. Copy `compare_models.py` to your `conversion-analyzer-starter/` folder alongside `train.py`.
2. Keep the existing dataset at `data/online_shoppers_intention.csv`.
3. In VS Code Terminal, from the project root, run:

   ```powershell
   .\.venv\Scripts\python.exe compare_models.py
   ```

4. Read the printed metrics for both models.
5. Open `reports/pagevalues_model_comparison.png` for your presentation.
6. Keep `models/purchase_model.joblib`, `train.py`, `agent.py`, and the live Streamlit application unchanged.

## Important interpretation

- Both experiments use exactly the same 80/20 stratified train/test split and Random Forest settings as `train.py`.
- Compare ROC-AUC, recall, F1, precision, accuracy. Accuracy alone is misleading due to the imbalanced dataset.
- Do **not** claim that removing PageValues proves data leakage. The comparison is a robustness/sensitivity check.
- If performance declines, say that the model relied on information supplied by PageValues, not that PageValues "caused" purchases.
- No model results should be invented; use the values printed on your own machine.
- The full dashboard must use actual session-level counts and rates, not checkout-stage or cart-abandonment numbers.

## Collaboration and timeboxing

- Your colleague's dashboard code belongs in `pages/02_Dashboard.py` (optional `pages/03_Recommendations.py`).
- Their dashboard should use `data/online_shoppers_intention.csv` and remain independent of the agent page.
- Avoid merging their branch directly into `main` until you have tested the pages on your laptop.
- Prepare the core technical slides in parallel: architecture, model evaluation, agent workflow, live demonstration.
