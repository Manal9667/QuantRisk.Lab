"""Construct a ``PortfolioProblem`` from a daily-return matrix.

Bridges the data/DB layer (a returns DataFrame + sector map) and the optimization
layers (a fully specified numerical problem).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantumrisklab.domain import PortfolioProblem
from quantumrisklab.features.returns import (
    annualise_cov,
    annualise_returns,
    daily_covariance,
    daily_mean_returns,
)
from quantumrisklab.features.risk import asset_betas, market_proxy_returns

TRADING_DAYS_PER_YEAR = 252


def build_problem_from_frame(
    returns: pd.DataFrame,
    sectors: dict[str, str],
    prev_weights: Optional[np.ndarray] = None,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    tickers: Optional[Sequence[str]] = None,
) -> PortfolioProblem:
    """Build a ``PortfolioProblem`` from a (dates x tickers) returns frame.

    ``sectors`` maps ticker -> sector. If ``tickers`` is given, the problem is
    restricted (and ordered) to that subset.
    """
    if returns.empty:
        raise ValueError("returns frame is empty")

    if tickers is not None:
        missing = [t for t in tickers if t not in returns.columns]
        if missing:
            raise ValueError(f"tickers not present in returns frame: {missing}")
        returns = returns[list(tickers)]

    ordered_tickers = list(returns.columns)
    r = returns.to_numpy(dtype=float)

    daily_mean = daily_mean_returns(r)
    d_cov = daily_covariance(r)
    exp_returns = annualise_returns(daily_mean, trading_days)
    cov = annualise_cov(d_cov, trading_days)

    market = market_proxy_returns(r)
    betas = asset_betas(r, market)

    sector_list = [sectors.get(t, "Unknown") for t in ordered_tickers]

    return PortfolioProblem(
        tickers=ordered_tickers,
        sectors=sector_list,
        expected_returns=exp_returns,
        cov_matrix=cov,
        betas=betas,
        daily_mean=daily_mean,
        daily_cov=d_cov,
        returns_matrix=r,
        market_returns=market,
        prev_weights=prev_weights,
        trading_days=trading_days,
    )
