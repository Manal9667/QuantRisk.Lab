"""Central configuration for the Credit Risk Stress Testing Engine.

Every tunable assumption lives here so that the distinction between
*observed data*, *model outputs* and *analyst assumptions* stays explicit and
auditable. Nothing in this file is a regulatory standard - the thresholds and
loss parameters are transparent modelling choices for a prototype.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATASET_PATH = RAW_DATA_DIR / "credit_risk_dataset.csv"
MODEL_PATH = MODELS_DIR / "pd_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# --------------------------------------------------------------------------- #
# Target and features
# --------------------------------------------------------------------------- #
TARGET = "loan_status"  # 1 = default event observed, 0 = no default

# Features known at loan origination / scoring time (no post-outcome leakage).
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Column used as the Exposure At Default (EAD) proxy. This dataset has no
# explicit EAD field, so we use the loan amount (see EAD_NOTE in README /
# expected_loss.py).
EAD_COLUMN = "loan_amnt"

# --------------------------------------------------------------------------- #
# Data cleaning bounds (guard against obvious data-entry errors)
# --------------------------------------------------------------------------- #
# The raw file contains implausible records such as age = 144 and employment
# length = 123 years. We treat these as invalid and drop them.
MAX_PLAUSIBLE_AGE = 100
MAX_PLAUSIBLE_EMP_LENGTH = 60

# --------------------------------------------------------------------------- #
# Risk / loss assumptions
# --------------------------------------------------------------------------- #
# Loss Given Default. 45% is a common textbook "senior unsecured" placeholder
# (and echoes the Basel foundation-IRB downturn LGD figure). It is an
# ASSUMPTION here, not an estimate from data, and is configurable in the UI.
DEFAULT_LGD = 0.45

# Risk-band thresholds on Probability of Default (PD). These are documented
# modelling choices, NOT an industry/regulatory standard.
RISK_BANDS = [
    ("Low Risk", 0.00, 0.05),
    ("Moderate Risk", 0.05, 0.15),
    ("High Risk", 0.15, 0.30),
    ("Very High Risk", 0.30, 1.01),
]
RISK_BAND_ORDER = [b[0] for b in RISK_BANDS]

# Exposure in the "High Risk" or "Very High Risk" bands is reported as
# "high-risk exposure".
HIGH_RISK_BANDS = ["High Risk", "Very High Risk"]

# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.25
RANDOM_STATE = 42
