"""Generates notebooks/analysis.ipynb programmatically.

Run once (``python notebooks/build_notebook.py``) to (re)build the walkthrough
notebook. Kept as a script so the notebook content is version-controllable and
reproducible. The notebook is then executed with nbconvert to embed real
outputs.
"""

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Credit Risk Stress Testing Engine — Analysis Walkthrough

This notebook walks through the end-to-end methodology exactly as implemented in
the `src/` package:

1. Load & validate a real public credit-risk dataset
2. Clean and profile the data
3. Train a Probability of Default (PD) model (+ compare & calibrate)
4. Evaluate with ranking **and** calibration metrics
5. Compute Expected Credit Loss (EL = PD × LGD × EAD)
6. Segment the portfolio into risk bands
7. Run hypothetical stress scenarios

> **Reminders:** Expected Loss is a forward-looking *statistical expectation*, not a
> realised loss. Stress scenarios are *hypothetical sensitivity analyses*, not
> forecasts. The LGD assumption and risk-band thresholds are transparent modelling
> choices, not regulatory standards.""")

code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
pd.set_option('display.float_format', lambda v: f'{v:,.4f}')

from src.data_loader import load_raw_data, profile_data
from src.preprocessing import clean_data, split_data""")

md("## 1–2. Load, validate, clean and profile")

code("""raw = load_raw_data()
print(profile_data(raw).summary())""")

code("""clean = clean_data(raw)
print(f'Rows: {len(raw):,} raw -> {len(clean):,} after removing implausible records')
clean.head()""")

md("""### Train/test split (stratified on the imbalanced default target)
Preprocessing (imputation, scaling, one-hot encoding) happens *inside* the model
pipeline so statistics are learned from the training fold only — this prevents
leakage.""")

code("""split = split_data(clean)
print(f'Train: {split.train_size:,}  |  Test: {split.test_size:,}')
print(f'Train default rate: {split.y_train.mean():.2%}  |  Test default rate: {split.y_test.mean():.2%}')""")

md("## 3–4. Train the PD model, compare and calibrate")

code("""from src.pd_model import train

model, report = train(split, compare_gbm=True, calibrate=True)
comp = pd.DataFrame({k: {m: v[m] for m in ['roc_auc','pr_auc','precision','recall','f1','brier_score']}
                     for k, v in report['models'].items()}).T
print('Chosen model:', report['chosen_model'])
comp""")

md("""**Why calibration matters here:** PD feeds *directly* into Expected Loss
(EL = PD × LGD × EAD). A model can rank borrowers well (high ROC-AUC) yet produce
mis-scaled probabilities, which would bias every loss number. We therefore select
the variant with the lower **Brier score** (better-calibrated probabilities).""")

code("""# Calibration curve for the deployed model
import matplotlib.pyplot as plt

pick = report['models'].get('logistic_calibrated') or report['models']['logistic_uncalibrated']
cal = pick['calibration_curve']
plt.figure(figsize=(5,5))
plt.plot(cal['mean_predicted'], cal['fraction_positive'], 'o-', label='Model')
plt.plot([0,1],[0,1],'--',color='grey', label='Perfectly calibrated')
plt.xlabel('Mean predicted PD'); plt.ylabel('Observed default frequency')
plt.title(f"Calibration (Brier={pick['brier_score']:.4f})"); plt.legend(); plt.show()""")

md("## 5. Expected Credit Loss (EL = PD × LGD × EAD)")

code("""from src.expected_loss import build_loss_table, portfolio_metrics
from src.risk_segmentation import add_risk_bands, segment_summary
from src import config

pd_hat = model.predict_pd(clean)
loss_table = add_risk_bands(build_loss_table(clean, pd_hat, lgd=config.DEFAULT_LGD))
pm = portfolio_metrics(loss_table)

print(f'LGD assumption      : {config.DEFAULT_LGD:.0%}')
print(f'Total EAD           : ${pm.total_ead:,.0f}')
print(f'Average PD (exp-wtd): {pm.average_pd:.2%}')
print(f'Total Expected Loss : ${pm.total_expected_loss:,.0f}')
print(f'Expected Loss rate  : {pm.expected_loss_rate:.2%}')
print(f'High-risk exposure  : ${pm.high_risk_exposure:,.0f} ({pm.high_risk_exposure_pct:.1%})')""")

md("## 6. Risk segmentation")

code("""segment_summary(loss_table)[['risk_band','n_borrowers','total_ead','average_pd',
                             'expected_loss','exposure_pct']]""")

md("## 7. Stress testing (hypothetical sensitivity analysis)")

code("""from src.stress_testing import run_all_scenarios, scenario_comparison_table

results = run_all_scenarios(clean, model, lgd=config.DEFAULT_LGD)
scenario_comparison_table(results)[['scenario','average_pd','expected_loss',
                                    'expected_loss_rate','el_increase_vs_base',
                                    'pct_increase_vs_base']]""")

code("""comp = scenario_comparison_table(results)
plt.figure(figsize=(6,4))
plt.bar(comp['scenario'], comp['expected_loss'], color=['#1f77b4','#ff7f0e','#d62728'])
plt.ylabel('Expected Loss ($)'); plt.title('Portfolio Expected Loss by scenario')
plt.show()""")

md("""## Takeaways

- The logistic PD model ranks default risk well (ROC-AUC) and, after isotonic
  calibration, produces better-scaled probabilities (lower Brier) — important
  because PD drives every loss figure.
- Expected Loss concentrates in the high-PD bands and in the weaker loan grades.
- Under the hypothetical **Severe** scenario, portfolio Expected Loss increases
  materially versus the base case — this is a *sensitivity* result, not a forecast.

See the Streamlit dashboard (`streamlit run dashboard/app.py`) for the interactive
version, including borrower-level explainability and model diagnostics.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}

with open(os.path.join(os.path.dirname(__file__), 'analysis.ipynb'), 'w') as f:
    nbf.write(nb, f)
print('Wrote analysis.ipynb')
