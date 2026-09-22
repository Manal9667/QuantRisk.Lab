"""Unit tests for the stress-scenario transformations."""

import numpy as np
import pandas as pd
import pytest

from src.stress_testing import (
    DEFAULT_SCENARIOS,
    StressScenario,
    apply_scenario,
)


def _borrowers():
    return pd.DataFrame(
        {
            "person_age": [30, 45],
            "person_income": [100_000.0, 60_000.0],
            "person_emp_length": [5.0, 2.0],
            "loan_amnt": [20_000.0, 12_000.0],
            "loan_int_rate": [10.0, 14.0],
            "loan_percent_income": [0.20, 0.20],
            "cb_person_cred_hist_length": [8, 4],
            "person_home_ownership": ["RENT", "OWN"],
            "loan_intent": ["PERSONAL", "MEDICAL"],
            "loan_grade": ["B", "C"],
            "cb_person_default_on_file": ["N", "Y"],
        }
    )


def test_base_scenario_is_identity():
    df = _borrowers()
    out = apply_scenario(df, DEFAULT_SCENARIOS["Base"])
    pd.testing.assert_frame_equal(df, out)


def test_base_scenario_does_not_mutate_input():
    df = _borrowers()
    original = df.copy()
    _ = apply_scenario(df, DEFAULT_SCENARIOS["Moderate"])
    pd.testing.assert_frame_equal(df, original)


def test_income_shock_reduces_income():
    df = _borrowers()
    sc = StressScenario(name="t", description="", income_shock=0.75)
    out = apply_scenario(df, sc)
    np.testing.assert_allclose(out["person_income"], df["person_income"] * 0.75)


def test_rate_shock_is_additive():
    df = _borrowers()
    sc = StressScenario(name="t", description="", rate_shock_pp=5.0)
    out = apply_scenario(df, sc)
    np.testing.assert_allclose(out["loan_int_rate"], df["loan_int_rate"] + 5.0)


def test_dti_increases_when_income_falls():
    df = _borrowers()
    sc = StressScenario(name="t", description="", income_shock=0.5)
    out = apply_scenario(df, sc)
    # Lower income mechanically raises loan-to-income ratio.
    assert (out["loan_percent_income"] > df["loan_percent_income"]).all()


def test_emp_length_shock_floored_at_zero():
    df = _borrowers()
    sc = StressScenario(name="t", description="", emp_length_shock_years=10.0)
    out = apply_scenario(df, sc)
    assert (out["person_emp_length"] >= 0).all()


def test_severe_is_harsher_than_moderate_on_income():
    df = _borrowers()
    moderate = apply_scenario(df, DEFAULT_SCENARIOS["Moderate"])
    severe = apply_scenario(df, DEFAULT_SCENARIOS["Severe"])
    assert (severe["person_income"] < moderate["person_income"]).all()
    assert (severe["loan_int_rate"] > moderate["loan_int_rate"]).all()


def test_exposure_is_held_constant():
    df = _borrowers()
    out = apply_scenario(df, DEFAULT_SCENARIOS["Severe"])
    np.testing.assert_allclose(out["loan_amnt"], df["loan_amnt"])
