"""Deterministic synthetic market data generation.

We deliberately avoid live market feeds so the pipeline is fully reproducible and
offline. Prices are simulated from a linear factor model:

    r_{i,t} = alpha_i
              + beta_i * f^market_t
              + gamma_i * f^{sector(i)}_t
              + eps_{i,t}

where the market and sector factors and idiosyncratic noise are Gaussian. This
produces a realistic covariance structure (systematic + sector + idiosyncratic
risk) and genuine cross-sectional dispersion in betas, which is exactly what the
optimizers and risk metrics need to be meaningful. Everything is driven by a
single seed via ``numpy.random.default_rng`` so runs are bit-for-bit repeatable.

No financial results are hardcoded: only *generative parameters* (drifts, vols,
factor loadings) are set; all returns, covariances, and metrics are computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A fixed sector taxonomy (roughly mirrors GICS-style sectors a bank would use).
SECTORS: tuple[str, ...] = (
    "Financials",
    "Technology",
    "Healthcare",
    "Energy",
    "ConsumerStaples",
    "ConsumerDiscretionary",
    "Industrials",
    "Utilities",
    "Materials",
    "RealEstate",
)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class SyntheticUniverse:
    """A generated investable universe with a price history."""

    tickers: list[str]
    names: list[str]
    sectors: list[str]
    dates: pd.DatetimeIndex
    prices: pd.DataFrame  # index=dates, columns=tickers, adjusted close


def _annual_to_daily_drift(annual: float) -> float:
    return annual / TRADING_DAYS_PER_YEAR


def _annual_to_daily_vol(annual: float) -> float:
    return annual / np.sqrt(TRADING_DAYS_PER_YEAR)


def generate_universe(
    n_assets: int,
    n_trading_days: int,
    seed: int,
    start_price: float = 100.0,
) -> SyntheticUniverse:
    """Generate ``n_assets`` price series over ``n_trading_days`` business days.

    Parameters are chosen to yield plausible annualised equity statistics
    (single-digit to low-double-digit drifts, 15-40% vols).
    """
    if not 1 <= n_assets <= 500:
        raise ValueError("n_assets must be in [1, 500]")
    if n_trading_days < 30:
        raise ValueError("n_trading_days must be >= 30")

    rng = np.random.default_rng(seed)

    # --- asset static attributes -----------------------------------------
    sectors = [SECTORS[i % len(SECTORS)] for i in range(n_assets)]
    tickers = [f"AST{i:03d}" for i in range(n_assets)]
    names = [f"Synthetic {sectors[i]} Co. {i:03d}" for i in range(n_assets)]

    # Factor loadings.
    market_beta = rng.normal(1.0, 0.30, size=n_assets).clip(0.1, 2.5)
    sector_beta = rng.normal(0.6, 0.20, size=n_assets).clip(0.05, 1.5)
    # Small idiosyncratic annual alpha, mostly noise around zero.
    alpha_annual = rng.normal(0.01, 0.03, size=n_assets)
    # Idiosyncratic annual volatility.
    idio_vol_annual = rng.uniform(0.12, 0.35, size=n_assets)

    # --- factor time series ----------------------------------------------
    market_drift_annual = 0.07
    market_vol_annual = 0.16
    market_factor = rng.normal(
        _annual_to_daily_drift(market_drift_annual),
        _annual_to_daily_vol(market_vol_annual),
        size=n_trading_days,
    )

    unique_sectors = sorted(set(sectors))
    sector_vol_annual = 0.10
    sector_factors = {
        s: rng.normal(0.0, _annual_to_daily_vol(sector_vol_annual), size=n_trading_days)
        for s in unique_sectors
    }

    # --- asset daily returns ---------------------------------------------
    daily_alpha = np.array([_annual_to_daily_drift(a) for a in alpha_annual])
    daily_idio_vol = np.array([_annual_to_daily_vol(v) for v in idio_vol_annual])

    returns = np.empty((n_trading_days, n_assets), dtype=float)
    for i in range(n_assets):
        sector_series = sector_factors[sectors[i]]
        idio = rng.normal(0.0, daily_idio_vol[i], size=n_trading_days)
        returns[:, i] = (
            daily_alpha[i]
            + market_beta[i] * market_factor
            + sector_beta[i] * sector_series
            + idio
        )

    # --- prices from returns ---------------------------------------------
    price_paths = start_price * np.cumprod(1.0 + returns, axis=0)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_trading_days)
    prices = pd.DataFrame(price_paths, index=dates, columns=tickers)

    return SyntheticUniverse(
        tickers=tickers,
        names=names,
        sectors=sectors,
        dates=dates,
        prices=prices,
    )
