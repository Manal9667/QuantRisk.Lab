"""Classical portfolio optimizer (SciPy).

Cardinality-constrained mean-variance optimization is a mixed-integer quadratic
program (MIQP). SciPy has no integer solver, so cardinality is handled with a
transparent two-stage heuristic:

  Stage 1 (select): solve the continuous problem over the full universe with all
      *convex* constraints (budget, box, volatility, sector, beta). Rank assets
      by resulting weight.
  Stage 2 (weight): if the number of held names exceeds ``max_holdings``, keep the
      top-K names and re-solve weights over just that subset, now enforcing the
      minimum position size.

This is a heuristic, not a global MIQP solve. That is stated honestly in the
README; it is a strong, fast classical baseline and mirrors common practitioner
workflows. The identical Stage-2 weighting is reused to weight quantum-selected
subsets, isolating the *selection* decision for the comparison.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from quantumrisklab.domain import (
    OptimizationResult,
    PortfolioConstraints,
    PortfolioProblem,
)
from quantumrisklab.features.risk import portfolio_metrics
from quantumrisklab.optimization.objective import portfolio_utility, solve_weights

SELECTION_THRESHOLD = 1e-4


def optimize_classical(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    risk_free_rate: float = 0.02,
    seed: Optional[int] = None,
    compute_metrics: bool = True,
) -> OptimizationResult:
    """Run the two-stage classical optimizer and return an OptimizationResult."""
    start = time.perf_counter()
    n = problem.n

    # Stage 1: continuous solve over the full universe (no min-weight floor yet).
    w1 = solve_weights(problem, constraints, subset=None, enforce_min_weight=False, seed=seed)
    held = np.where(w1 > SELECTION_THRESHOLD)[0]

    # Stage 2: enforce cardinality if required.
    if constraints.max_holdings is not None and len(held) > constraints.max_holdings:
        top_k = held[np.argsort(w1[held])[::-1][: constraints.max_holdings]]
        subset = np.sort(top_k)
    else:
        subset = np.sort(held) if len(held) > 0 else np.arange(n)

    weights = solve_weights(problem, constraints, subset=subset, enforce_min_weight=True, seed=seed)
    selected = weights > SELECTION_THRESHOLD

    objective_value = portfolio_utility(weights, problem, constraints)
    runtime = time.perf_counter() - start

    metrics = (
        portfolio_metrics(weights, problem, constraints, risk_free_rate)
        if compute_metrics
        else None
    )

    return OptimizationResult(
        run_type="classical",
        solver="scipy_slsqp_two_stage",
        tickers=list(problem.tickers),
        weights=weights,
        selected=selected,
        objective_value=objective_value,
        runtime_seconds=runtime,
        universe_size=n,
        n_variables=n,  # continuous weight variables
        metrics=metrics,
        notes=(
            f"two-stage SLSQP; stage1_held={len(held)}; "
            f"max_holdings={constraints.max_holdings}"
        ),
    )
