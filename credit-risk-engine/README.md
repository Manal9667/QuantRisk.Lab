# Credit Risk Stress Testing Engine

A compact, end-to-end **credit-risk analytics** tool that estimates each borrower's
Probability of Default (PD), converts it into **Expected Credit Loss**
(EL = PD × LGD × EAD), segments the book into risk bands, and — the centrepiece —
runs **hypothetical macro-stress scenarios** to show how portfolio losses respond
when the economy deteriorates. Everything is presented through a polished Streamlit
dashboard. It is built to look like something a **credit-risk / risk-analytics
intern** would prototype, not a generic "loan default" ML demo.

> **Prototype, not a production model.** Assumptions are explicit and configurable.
> Nothing here is a regulatory standard, and the stress scenarios are sensitivity
> analyses — **not economic forecasts**.

---

## Problem

Lenders hold portfolios of loans that can lose value when borrowers default. Two
questions matter to a risk team:

1. **How much loss should we *expect* on this book today?** — the Expected Credit
   Loss, driven by each borrower's probability of default.
2. **How bad could it get if conditions worsen?** — incomes fall, borrowers become
   more leveraged, rates rise. This is what **stress testing** answers.

This project builds a transparent pipeline for both, on a real public dataset.

---

## Architecture

```
        Raw data  (public credit-risk dataset, ~32.6k loans)
           │
           ▼
     Preprocessing  ── validation • clean implausible records • impute • scale • one-hot
           │           (leakage-safe: fit inside the model pipeline, on train fold only)
           ▼
        PD Model  ── Logistic Regression (baseline) vs Gradient Boosting (benchmark)
           │
           ▼
     PD Calibration ── isotonic CalibratedClassifierCV, chosen by Brier score
           │
           ▼
     PD × LGD × EAD ── Expected Credit Loss per borrower
           │
           ▼
   Risk Segmentation ── Low / Moderate / High / Very-High PD bands
           │
           ▼
    Stress Testing ── Base / Moderate / Severe hypothetical scenarios
           │
           ▼
       Dashboard  ── Streamlit: KPIs • scenarios • concentration • borrower drivers • diagnostics
```

### Repository layout

```
credit-risk-engine/
├── data/
│   ├── raw/credit_risk_dataset.csv     # the public dataset (tracked so it runs out-of-the-box)
│   └── processed/                      # regenerated artefacts (git-ignored)
├── src/
│   ├── config.py            # all tunable assumptions live here
│   ├── data_loader.py       # load + schema validation + data profile
│   ├── preprocessing.py     # cleaning, ColumnTransformer, stratified split
│   ├── pd_model.py          # PD model, calibration, evaluation, persistence
│   ├── expected_loss.py     # EL = PD × LGD × EAD engine + portfolio aggregation
│   ├── risk_segmentation.py # PD-based risk bands
│   ├── stress_testing.py    # scenario transforms + portfolio re-scoring
│   └── explainability.py    # borrower-level log-odds contributions
├── dashboard/app.py         # Streamlit dashboard
├── notebooks/analysis.ipynb # executed methodology walkthrough
├── tests/                   # pytest unit tests
├── models/                  # persisted model + metrics (regenerated)
├── requirements.txt
└── README.md
```

---

## Dataset

**"Credit Risk Dataset"** — a public consumer-loan dataset (originally distributed on
Kaggle as `credit_risk_dataset.csv`; this copy was retrieved from a public GitHub
mirror). ~32,581 loan records with borrower, loan and credit-bureau attributes and a
binary default outcome.

| Field | Meaning | Role |
|---|---|---|
| `person_age` | Applicant age | feature |
| `person_income` | Annual income | feature (stressed) |
| `person_emp_length` | Employment length (years) | feature (stressed) |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER | feature |
| `loan_intent` | Purpose (personal, education, medical, …) | feature / concentration dim |
| `loan_grade` | Credit grade A–G | feature / concentration dim |
| `loan_amnt` | Loan amount | **EAD proxy** |
| `loan_int_rate` | Interest rate (%) | feature (stressed) |
| `loan_percent_income` | Loan ÷ income (leverage / DTI proxy) | feature (stressed) |
| `cb_person_default_on_file` | Prior default on file (Y/N) | feature |
| `cb_person_cred_hist_length` | Credit-history length (years) | feature |
| `loan_status` | **1 = default, 0 = no default** | **target** |

**Observed data facts** (computed, not fabricated): 32,581 rows; **21.82% default
rate**; missing values in `person_emp_length` (895) and `loan_int_rate` (3,116).
Records with implausible values (age > 100, employment length > 60y) are dropped in
cleaning, leaving **32,574** rows.

### Assumptions vs. observed data vs. model output

The app is careful to distinguish these:

- **Observed data:** default rate, feature distributions, exposure amounts.
- **Model output:** PD, Expected Loss, risk bands.
- **Assumptions:** LGD, the EAD proxy, and risk-band thresholds (below).
- **Hypothetical:** the stress-scenario shock magnitudes.

**EAD assumption.** The dataset has no explicit *Exposure At Default* field, so EAD is
proxied by `loan_amnt`. This assumes the full drawn principal is outstanding at
default — a transparent, slightly conservative simplification for an amortising loan.

**LGD assumption.** *Loss Given Default* is not in the data. We use **45%** by default
(a common senior-unsecured placeholder that echoes the Basel foundation-IRB figure).
It is an assumption, **not** an estimate, and is a slider in the dashboard.

---

## Risk methodology

- **PD — Probability of Default.** `P(default | borrower characteristics)`, produced
  by the model and calibrated so the numbers are usable as probabilities.
- **LGD — Loss Given Default.** Fraction of exposure lost if default occurs
  (assumption, default 45%).
- **EAD — Exposure At Default.** Money at risk when default occurs (proxied by loan
  amount).
- **EL — Expected Credit Loss.**

  ```
  EL = PD × LGD × EAD
  ```

  Aggregated across borrowers to get total portfolio EL, EL rate (EL ÷ EAD),
  exposure-weighted average PD, and high-risk exposure.

> **Expected Loss ≠ realised loss.** EL is a forward-looking *statistical
> expectation* given the modelled default probability, not money actually lost by any
> borrower. The app states this explicitly.

**Risk bands** (fixed PD thresholds — a modelling choice, *not* a regulatory
standard):

| Band | PD range |
|---|---|
| Low Risk | < 5% |
| Moderate Risk | 5% – 15% |
| High Risk | 15% – 30% |
| Very High Risk | ≥ 30% |

---

## Stress testing

Three scenarios apply hypothetical deterioration to borrower characteristics, then the
model **re-scores PD** and Expected Loss is recomputed. Exposure (loan amount) is held
constant — we stress creditworthiness, not loan sizing.

| Scenario | Income | Interest rate | Leverage / DTI | Job tenure |
|---|---|---|---|---|
| **Base** | — | — | — | — |
| **Moderate** | −10% | +200 bps | +10% | −1 yr |
| **Severe** | −25% | +500 bps | +25% | −3 yr |

For each scenario the app reports average PD, total EAD, Expected Loss, EL rate,
high-risk exposure, and the change vs. base:

```
Stress Loss Increase = Stress_EL − Base_EL
Percentage Increase  = (Stress_EL / Base_EL − 1) × 100
```

> **These are hypothetical sensitivity analyses, not economic forecasts.** The shock
> magnitudes are illustrative analyst assumptions and are configurable in
> `src/stress_testing.py`.

---

## Model evaluation

Metrics below are **computed by this project** on a held-out, **stratified** 25% test
fold (24,430 train / 8,144 test; test default rate 21.82%). They are not fabricated —
re-run `python -m src.pd_model` to reproduce.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression (uncalibrated) | 0.860 | 0.703 | 0.543 | 0.754 | 0.631 | 0.1408 |
| Logistic Regression (**isotonic-calibrated**) ✅ deployed | 0.860 | 0.700 | 0.728 | 0.571 | 0.640 | **0.1059** |
| Gradient Boosting (benchmark) | 0.919 | 0.864 | 0.948 | 0.694 | 0.801 | 0.0662 |

**Why calibration is emphasised.** PD feeds *directly* into Expected Loss, so the
*level* of the probabilities matters, not just their ranking. Isotonic calibration
improved the logistic model's **Brier score from 0.1408 → 0.1059** (lower is better)
while preserving ranking (ROC-AUC 0.860). The calibrated logistic model is deployed as
the PD scorer.

**Why not just use the gradient-boosting model?** It ranks higher on ROC/PR-AUC, and
we keep it transparently as a benchmark. For this prototype we deploy logistic
regression because it is transparent, naturally interpretable via coefficients, and
calibrates cleanly — properties a risk team values for a PD scorecard. Accuracy is
deliberately **not** the headline metric: with a ~22% default rate it would reward a
lazy "no default" prediction.

**Portfolio result** (deployed model, LGD = 45%, EAD = loan amount): total exposure
≈ **$312M**, exposure-weighted average PD ≈ **24.5%**, total Expected Loss ≈ **$34.5M**
(EL rate ≈ **11.0%**), with ≈ **39%** of exposure in the High / Very-High bands.
Under the **Severe** scenario, Expected Loss roughly **doubles** versus base
(a sensitivity result, not a prediction). Exact figures are shown live in the
dashboard.

---

## Explainability

For a logistic model the exact, clean attribution is the **log-odds contribution** of
each feature for a borrower: `coef_i × x_i`. The borrower view ranks the features
pushing PD up vs. down and uses **associational** language — *"associated with higher
predicted PD"* — never causal claims like *"causes default."*

---

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · SciPy · Plotly / Matplotlib ·
Streamlit · Joblib · pytest. (SHAP is listed as optional; coefficient contributions
are exact for the linear model and are used by default.)

---

## Running locally

```bash
# from the credit-risk-engine/ directory
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train + calibrate the PD model and write models/pd_model.joblib + metrics
python -m src.pd_model

# 2) Launch the dashboard
streamlit run dashboard/app.py

# 3) Run the tests
pytest -q

# (optional) rebuild the executed analysis notebook
python notebooks/build_notebook.py
```

Open the dashboard at the URL Streamlit prints (default http://localhost:8501).

---

## Limitations

- **Dataset.** A single public dataset of unknown vintage/geography; `loan_grade` and
  `loan_int_rate` are themselves risk-derived, so the model partly learns an existing
  underwriting view rather than pure fundamentals.
- **LGD / EAD assumptions.** LGD is a flat 45% assumption, not estimated from
  recoveries; EAD is proxied by loan amount. Real ECL uses collateral, seniority and
  exposure modelling.
- **Hypothetical stress.** Scenario shocks are illustrative analyst assumptions applied
  uniformly to all borrowers — **not** macro-econometric forecasts, and they ignore
  correlations between shocks.
- **Model.** A logistic baseline on cross-sectional data; no time dimension, no
  through-the-cycle vs. point-in-time treatment, no out-of-time validation.
- **Not production / not regulatory.** This is a learning prototype. It makes **no**
  claim of regulatory compliance (IFRS 9, CECL, Basel) and is not equivalent to a
  bank's validated production risk model.
