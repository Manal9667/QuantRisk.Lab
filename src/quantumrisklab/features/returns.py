"""Return-based feature calculation.

Expected returns and covariances are estimated from the daily return matrix and
annualised. These are inputs to both the classical optimizer and the QUBO.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def daily_mean_returns(returns: np.ndarray) -> np.ndarray:
    """Mean daily return per asset. ``returns`` has shape (T, n)."""
    return np.asarray(returns, dtype=float).mean(axis=0)


def daily_covariance(returns: np.ndarray) -> np.ndarray:
    """Sample covariance of daily returns, shape (n, n)."""
    r = np.asarray(returns, dtype=float)
    if r.shape[0] < 2:
        raise ValueError("Need at least 2 observations to estimate covariance.")
    # rowvar=False: variables are columns (assets).
    return np.cov(r, rowvar=False, ddof=1)


def annualise_returns(
    daily_mean: np.ndarray, trading_days: int = TRADING_DAYS_PER_YEAR
) -> np.ndarray:
    """Annualise mean daily returns (arithmetic annualisation)."""
    return np.asarray(daily_mean, dtype=float) * trading_days


def annualise_cov(
    daily_cov: np.ndarray, trading_days: int = TRADING_DAYS_PER_YEAR
) -> np.ndarray:
    """Annualise a daily covariance matrix."""
    return np.asarray(daily_cov, dtype=float) * trading_days
