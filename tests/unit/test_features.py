"""Unit tests for return and risk feature calculation."""

from __future__ import annotations

import numpy as np

from quantumrisklab.features.returns import (
    annualise_cov,
    annualise_returns,
    daily_covariance,
    daily_mean_returns,
)
from quantumrisklab.features.risk import (
    asset_betas,
    historical_var_cvar,
    market_proxy_returns,
    max_drawdown,
    portfolio_metrics,
    rolling_volatility,
)


def test_annualisation():
    daily = np.array([0.001, 0.002])
    np.testing.assert_allclose(annualise_returns(daily, 252), daily * 252)
    cov = np.array([[1e-4, 0.0], [0.0, 2e-4]])
    np.testing.assert_allclose(annualise_cov(cov, 252), cov * 252)


def test_covariance_shape_symmetric(sample_problem):
    r = sample_problem.returns_matrix
    cov = daily_covariance(r)
    assert cov.shape == (r.shape[1], r.shape[1])
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert (np.diag(cov) >= 0).all()


def test_mean_returns_matches_numpy(sample_problem):
    r = sample_problem.returns_matrix
    np.testing.assert_allclose(daily_mean_returns(r), r.mean(axis=0))


def test_market_proxy_is_cross_sectional_mean(sample_problem):
    r = sample_problem.returns_matrix
    np.testing.assert_allclose(market_proxy_returns(r), r.mean(axis=1))


def test_beta_of_market_is_one():
    # If an asset IS the market, its beta must be ~1.
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 500)
    returns = np.column_stack([market, market * 2.0, rng.normal(0, 0.01, 500)])
    betas = asset_betas(returns, market)
    assert abs(betas[0] - 1.0) < 1e-6
    assert abs(betas[1] - 2.0) < 1e-6


def test_max_drawdown_sign():
    # A monotonically rising series has ~zero drawdown.
    up = np.full(50, 0.01)
    assert max_drawdown(up) >= -1e-9
    # A crash produces a negative drawdown.
    crash = np.array([0.1, -0.5, 0.0])
    assert max_drawdown(crash) < 0


def test_var_cvar_positive_losses():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, 1000)
    var, cvar = historical_var_cvar(r, 0.95)
    assert var > 0
    assert cvar >= var  # tail loss is at least the VaR threshold


def test_rolling_volatility_shape(sample_problem):
    r = sample_problem.returns_matrix
    rv = rolling_volatility(r, window=21)
    assert rv.shape == (r.shape[0] - 21 + 1, r.shape[1])
    assert (rv >= 0).all()


def test_portfolio_metrics_equal_weight(sample_problem, base_constraints):
    n = sample_problem.n
    w = np.full(n, 1.0 / n)
    m = portfolio_metrics(w, sample_problem, base_constraints)
    assert m.volatility > 0
    # Effective N of an equal-weight portfolio equals n.
    assert abs(m.effective_n - n) < 1e-6
    # Sector exposures sum to ~1.
    assert abs(sum(m.sector_exposure.values()) - 1.0) < 1e-9
