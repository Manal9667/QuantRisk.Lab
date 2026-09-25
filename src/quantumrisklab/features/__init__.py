"""Financial feature calculation: returns, risk statistics, and problem
construction."""

from __future__ import annotations

from quantumrisklab.features.problem import build_problem_from_frame
from quantumrisklab.features.returns import annualise_cov, annualise_returns
from quantumrisklab.features.risk import (
    asset_betas,
    max_drawdown,
    portfolio_metrics,
    rolling_volatility,
)

__all__ = [
    "annualise_cov",
    "annualise_returns",
    "asset_betas",
    "build_problem_from_frame",
    "max_drawdown",
    "portfolio_metrics",
    "rolling_volatility",
]
