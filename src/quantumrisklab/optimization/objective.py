"""Shared objective and the continuous weight solver.

The canonical portfolio *utility* is:

    U(w) = expected_return(w) - lambda * variance(w) - transaction_cost(w)

Both the classical optimizer and the quantum pipeline report ``U(w)`` as the
run's ``objective_value`` so classical and quantum solutions are compared on a
common footing. The QUBO energy is a *different* objective and is stored
separately on ``quantum_runs``.

``solve_weights`` performs continuous mean-variance weighting over a chosen
subset of assets using SciPy SLSQP. It is reused by:
  * the classical optimizer (weighting its selected names), and
  * the quantum pipeline (weighting the names the QUBO selected),
so that the *selection* decision is isolated from the *weighting* decision.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from quantumrisklab.domain import PortfolioConstraints, PortfolioProblem


def portfolio_utility(
    weights: np.ndarray,
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
) -> float:
    """Canonical utility U(w) (higher is better)."""
    w = np.asarray(weights, dtype=float)
    expected_return = float(w @ problem.expected_returns)
    variance = float(w @ problem.cov_matrix @ w)
    prev = problem.prev_weights if problem.prev_weights is not None else np.zeros(problem.n)
    tx_cost = constraints.transaction_cost * float(np.sum(np.abs(w - prev)))
    return expected_return - constraints.risk_aversion * variance - tx_cost


def solve_weights(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    subset: Optional[Sequence[int]] = None,
    enforce_min_weight: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Continuous mean-variance weighting over ``subset`` (indices into the
    universe). Assets outside the subset receive zero weight. Returns a
    full-length weight vector summing to ~1.
    """
    n = problem.n
    idx = np.arange(n) if subset is None else np.asarray(sorted(set(subset)), dtype=int)
    k = len(idx)
    if k == 0:
        return np.zeros(n)
    if k == 1:
        w = np.zeros(n)
        w[idx[0]] = 1.0
        return w

    mu = problem.expected_returns[idx]
    sigma = problem.cov_matrix[np.ix_(idx, idx)]
    betas = problem.betas[idx]
    sectors = [problem.sectors[i] for i in idx]
    prev_full = problem.prev_weights if problem.prev_weights is not None else np.zeros(n)
    prev = prev_full[idx]

    lam = constraints.risk_aversion
    tx = constraints.transaction_cost

    def objective(w: np.ndarray) -> float:
        expected = w @ mu
        variance = w @ sigma @ w
        tx_cost = tx * np.sum(np.abs(w - prev))
        return -(expected) + lam * variance + tx_cost

    # Bounds: lower is min_weight when enforcing (subset is "held"), else 0.
    lower = constraints.min_weight if enforce_min_weight else 0.0
    # Guard against infeasible lower bounds (k * min_weight must be <= 1).
    if lower * k > 1.0:
        lower = 1.0 / k
    upper = max(constraints.max_weight, lower)
    bounds = [(lower, upper)] * k

    cons: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if constraints.max_volatility is not None:
        vmax_sq = constraints.max_volatility**2
        cons.append({"type": "ineq", "fun": lambda w: vmax_sq - float(w @ sigma @ w)})

    if constraints.max_sector_exposure is not None:
        unique_sectors = sorted(set(sectors))
        sector_masks = {
            s: np.array([1.0 if sc == s else 0.0 for sc in sectors]) for s in unique_sectors
        }
        cap = constraints.max_sector_exposure
        for s in unique_sectors:
            mask = sector_masks[s]
            cons.append({"type": "ineq", "fun": (lambda w, m=mask: cap - float(w @ m))})

    if constraints.target_beta is not None and constraints.beta_tolerance is not None:
        tb = constraints.target_beta
        tol = constraints.beta_tolerance
        cons.append({"type": "ineq", "fun": lambda w: (tb + tol) - float(w @ betas)})
        cons.append({"type": "ineq", "fun": lambda w: float(w @ betas) - (tb - tol)})

    # Feasible warm start: equal weight (respecting bounds).
    x0 = np.full(k, 1.0 / k)
    x0 = np.clip(x0, lower, upper)
    if x0.sum() > 0:
        x0 = x0 / x0.sum()

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    w_sub = result.x if result.success else x0
    # Numerical clean-up: clip tiny negatives and renormalise.
    w_sub = np.clip(w_sub, 0.0, None)
    if w_sub.sum() > 0:
        w_sub = w_sub / w_sub.sum()

    w_full = np.zeros(n)
    w_full[idx] = w_sub
    return w_full
