"""Unit tests for the Expected Loss engine and risk-band assignment."""

import numpy as np
import pandas as pd
import pytest

from src.expected_loss import build_loss_table, expected_loss, portfolio_metrics
from src.risk_segmentation import assign_risk_band, assign_risk_bands, add_risk_bands


# --------------------------------------------------------------------------- #
# Core formula: EL = PD x LGD x EAD
# --------------------------------------------------------------------------- #
def test_expected_loss_reference_case():
    # The canonical worked example from the spec.
    assert expected_loss(0.10, 0.45, 100_000) == pytest.approx(4_500.0)


def test_expected_loss_zero_pd_is_zero():
    assert expected_loss(0.0, 0.45, 100_000) == 0.0


def test_expected_loss_vectorised():
    pd_vals = np.array([0.1, 0.2, 0.5])
    ead = np.array([100_000, 50_000, 10_000])
    result = expected_loss(pd_vals, 0.45, ead)
    np.testing.assert_allclose(result, [4_500.0, 4_500.0, 2_250.0])


def test_expected_loss_rejects_out_of_range_pd():
    with pytest.raises(ValueError):
        expected_loss(1.5, 0.45, 100_000)


def test_expected_loss_rejects_bad_lgd():
    with pytest.raises(ValueError):
        expected_loss(0.1, 1.5, 100_000)


def test_expected_loss_rejects_negative_ead():
    with pytest.raises(ValueError):
        expected_loss(0.1, 0.45, -100)


# --------------------------------------------------------------------------- #
# Portfolio aggregation
# --------------------------------------------------------------------------- #
def _toy_frame():
    return pd.DataFrame(
        {
            "loan_amnt": [100_000, 100_000],
            "pd_true": [0.10, 0.50],
        }
    )


def test_portfolio_metrics_totals():
    df = _toy_frame()
    lt = build_loss_table(df, df["pd_true"].values, lgd=0.45)
    lt = add_risk_bands(lt)
    pm = portfolio_metrics(lt)

    # EL = 0.10*0.45*100k + 0.50*0.45*100k = 4,500 + 22,500 = 27,000
    assert pm.total_expected_loss == pytest.approx(27_000.0)
    assert pm.total_ead == pytest.approx(200_000.0)
    assert pm.expected_loss_rate == pytest.approx(27_000.0 / 200_000.0)
    # Exposure-weighted average PD with equal exposure = simple mean = 0.30
    assert pm.average_pd == pytest.approx(0.30)
    assert pm.n_borrowers == 2


def test_high_risk_exposure_uses_bands():
    df = _toy_frame()
    lt = add_risk_bands(build_loss_table(df, df["pd_true"].values, lgd=0.45))
    pm = portfolio_metrics(lt)
    # PD=0.50 -> "Very High Risk" (high-risk); PD=0.10 -> "Moderate" (not).
    assert pm.high_risk_exposure == pytest.approx(100_000.0)


# --------------------------------------------------------------------------- #
# Risk-band assignment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pd_value,expected",
    [
        (0.00, "Low Risk"),
        (0.049, "Low Risk"),
        (0.05, "Moderate Risk"),
        (0.149, "Moderate Risk"),
        (0.15, "High Risk"),
        (0.299, "High Risk"),
        (0.30, "Very High Risk"),
        (0.99, "Very High Risk"),
        (1.00, "Very High Risk"),
    ],
)
def test_assign_risk_band_thresholds(pd_value, expected):
    assert assign_risk_band(pd_value) == expected


def test_assign_risk_band_rejects_out_of_range():
    with pytest.raises(ValueError):
        assign_risk_band(1.2)


def test_assign_risk_bands_vectorised_is_ordered():
    bands = assign_risk_bands([0.01, 0.10, 0.20, 0.80])
    assert list(bands) == ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
    assert bands.dtype.ordered is True
