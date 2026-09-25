"""Risk measurement.

Portfolio-level risk metrics computed from a weight vector and a
``PortfolioProblem``: volatility, Sharpe, max drawdown, beta, historical VaR/CVaR,
diversification ratio, effective number of holdings, turnover, transaction cost,
and sector exposure. Also exposes an equal-weighted market proxy and rolling
volatility used by the feature/analytics layers.
"""

from __future__ import annotations

import numpy as np

from quantumrisklab.domain import PortfolioConstraints, PortfolioMetrics, PortfolioProblem

TRADING_DAYS_PER_YEAR = 252


def market_proxy_returns(returns: np.ndarray) -> np.ndarray:
    """Equal-weighted market index return series from the asset return matrix.

    Using an equal-weighted index keeps the benchmark fully reconstructable from
    stored data (no external index needed) so betas are reproducible.
    """
    return np.asarray(returns, dtype=float).mean(axis=1)


def asset_betas(returns: np.ndarray, market: np.ndarray) -> np.ndarray:
    """Per-asset beta vs the market proxy: cov(r_i, r_m) / var(r_m)."""
    r = np.asarray(returns, dtype=float)
    m = np.asarray(market, dtype=float)
    var_m = np.var(m, ddof=1)
    if var_m == 0:
        return np.zeros(r.shape[1])
    # Covariance of each column with the market.
    m_centered = m - m.mean()
    r_centered = r - r.mean(axis=0, keepdims=True)
    cov_im = (r_centered * m_centered[:, None]).sum(axis=0) / (len(m) - 1)
    return cov_im / var_m


def rolling_volatility(
    returns: np.ndarray, window: int = 21, trading_days: int = TRADING_DAYS_PER_YEAR
) -> np.ndarray:
    """Annualised rolling volatility per asset.

    Returns an array of shape (T - window + 1, n). Provided for parity with the
    SQL ``rolling_volatility`` query and for dashboard use.
    """
    r = np.asarray(returns, dtype=float)
    if window < 2 or window > r.shape[0]:
        raise ValueError("window must be in [2, T]")
    T, n = r.shape
    out = np.empty((T - window + 1, n))
    for start in range(T - window + 1):
        chunk = r[start : start + window]
        out[start] = chunk.std(axis=0, ddof=1) * np.sqrt(trading_days)
    return out


def max_drawdown(portfolio_daily_returns: np.ndarray) -> float:
    """Maximum drawdown (a negative number) of a daily return series."""
    r = np.asarray(portfolio_daily_returns, dtype=float)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())


def historical_var_cvar(
    portfolio_daily_returns: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """1-day historical VaR and CVaR at the given confidence.

    Both are returned as positive loss magnitudes (e.g. 0.02 == 2% daily loss).
    """
    r = np.asarray(portfolio_daily_returns, dtype=float)
    if r.size == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    var_quantile = np.quantile(r, alpha)
    var = -float(var_quantile)
    tail = r[r <= var_quantile]
    cvar = -float(tail.mean()) if tail.size > 0 else var
    return var, cvar


def portfolio_metrics(
    weights: np.ndarray,
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    risk_free_rate: float = 0.02,
) -> PortfolioMetrics:
    """Compute the full risk/return snapshot for a weight vector."""
    w = np.asarray(weights, dtype=float)
    n = problem.n

    # Return / volatility (annualised).
    expected_return = float(w @ problem.expected_returns)
    variance = float(w @ problem.cov_matrix @ w)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0

    # Portfolio daily returns for path-dependent metrics.
    port_daily = problem.returns_matrix @ w
    mdd = max_drawdown(port_daily)
    var_95, cvar_95 = historical_var_cvar(port_daily, 0.95)

    # Beta of the portfolio.
    beta = float(w @ problem.betas)

    # Diversification ratio: weighted average vol / portfolio vol.
    asset_vols = np.sqrt(np.clip(np.diag(problem.cov_matrix), 0.0, None))
    weighted_avg_vol = float(w @ asset_vols)
    diversification_ratio = weighted_avg_vol / volatility if volatility > 0 else 0.0

    # Effective number of holdings (inverse Herfindahl).
    sum_sq = float(np.sum(w**2))
    effective_n = 1.0 / sum_sq if sum_sq > 0 else 0.0

    # Turnover / transaction cost vs previous weights.
    prev = problem.prev_weights if problem.prev_weights is not None else np.zeros(n)
    turnover = float(np.sum(np.abs(w - prev)))
    transaction_cost_paid = constraints.transaction_cost * turnover

    # Sector exposure.
    sector_exposure: dict[str, float] = {}
    for i, sector in enumerate(problem.sectors):
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + float(w[i])
    max_sector = max(sector_exposure.values()) if sector_exposure else 0.0

    violation = constraint_violation(w, problem, constraints)

    return PortfolioMetrics(
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        beta=beta,
        var_95=var_95,
        cvar_95=cvar_95,
        turnover=turnover,
        transaction_cost_paid=transaction_cost_paid,
        diversification_ratio=diversification_ratio,
        effective_n=effective_n,
        max_sector_exposure=max_sector,
        constraint_violation=violation,
        sector_exposure=sector_exposure,
    )


def constraint_violation(
    weights: np.ndarray,
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    tol: float = 1e-6,
) -> float:
    """Aggregate magnitude of constraint violations (0.0 == fully feasible).

    Sums the excess over each constraint so a single scalar summarises how far a
    solution is from feasibility. Used to score quantum solutions honestly.
    """
    w = np.asarray(weights, dtype=float)
    total = 0.0

    # Budget: weights should sum to ~1.
    total += abs(float(w.sum()) - 1.0)

    # Long-only / max weight.
    total += float(np.clip(-w, 0.0, None).sum())  # negative weights
    total += float(np.clip(w - constraints.max_weight, 0.0, None).sum())

    # Cardinality.
    if constraints.max_holdings is not None:
        held = int(np.count_nonzero(w > tol))
        total += max(0, held - constraints.max_holdings)

    # Volatility cap.
    if constraints.max_volatility is not None:
        vol = float(np.sqrt(max(w @ problem.cov_matrix @ w, 0.0)))
        total += max(0.0, vol - constraints.max_volatility)

    # Sector exposure cap.
    if constraints.max_sector_exposure is not None:
        sector_exposure: dict[str, float] = {}
        for i, sector in enumerate(problem.sectors):
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + float(w[i])
        for exposure in sector_exposure.values():
            total += max(0.0, exposure - constraints.max_sector_exposure)

    # Beta band.
    if constraints.target_beta is not None and constraints.beta_tolerance is not None:
        beta = float(w @ problem.betas)
        deviation = abs(beta - constraints.target_beta)
        total += max(0.0, deviation - constraints.beta_tolerance)

    return float(total)
