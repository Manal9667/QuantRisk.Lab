"""Application service layer.

Orchestrates: load data from DB -> build problem -> optimize -> persist. Used by
the FastAPI routes so the HTTP layer stays thin and the same logic is reusable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sqlalchemy.orm import Session

from quantumrisklab.config import get_settings
from quantumrisklab.db.repository import (
    create_constraint,
    load_returns_frame,
    persist_result,
    sector_map,
)
from quantumrisklab.domain import (
    OptimizationResult,
    PortfolioConstraints,
    PortfolioProblem,
    QuantumSolveDetail,
)
from quantumrisklab.features.problem import build_problem_from_frame
from quantumrisklab.optimization.classical import optimize_classical
from quantumrisklab.quantum.solver import optimize_quantum


def build_problem_for_request(
    session: Session,
    tickers: Optional[Sequence[str]] = None,
    universe_size: Optional[int] = None,
    prev_weights: Optional[np.ndarray] = None,
) -> PortfolioProblem:
    """Load returns from the DB and construct a PortfolioProblem.

    Selection precedence: explicit ``tickers`` > first ``universe_size`` assets >
    the full stored universe.
    """
    returns = load_returns_frame(session)
    if returns.empty:
        raise ValueError("No market data in the database. Seed it first (init_db --seed-data).")

    available = list(returns.columns)
    if tickers:
        chosen = list(tickers)
    elif universe_size:
        chosen = available[: int(universe_size)]
    else:
        chosen = available

    sectors = sector_map(session, chosen)
    return build_problem_from_frame(returns, sectors, prev_weights=prev_weights, tickers=chosen)


def _persist_constraint(session: Session, constraints: PortfolioConstraints) -> int:
    row = create_constraint(session, constraints)
    return row.constraint_id


def run_classical(
    session: Session,
    constraints: PortfolioConstraints,
    tickers: Optional[Sequence[str]] = None,
    universe_size: Optional[int] = None,
    experiment_id: Optional[int] = None,
    persist: bool = True,
    seed: Optional[int] = None,
) -> tuple[Optional[int], OptimizationResult]:
    settings = get_settings()
    problem = build_problem_for_request(session, tickers, universe_size)
    result = optimize_classical(
        problem, constraints, risk_free_rate=settings.risk_free_rate, seed=seed
    )
    run_id: Optional[int] = None
    if persist:
        constraint_id = _persist_constraint(session, constraints)
        run = persist_result(
            session,
            result,
            experiment_id=experiment_id,
            constraint_id=constraint_id,
            seed=seed,
        )
        run_id = run.run_id
    return run_id, result


def run_quantum(
    session: Session,
    constraints: PortfolioConstraints,
    backend: str = "exact",
    target_cardinality: Optional[int] = None,
    reps: int = 1,
    shots: int = 1024,
    tickers: Optional[Sequence[str]] = None,
    universe_size: Optional[int] = None,
    experiment_id: Optional[int] = None,
    persist: bool = True,
    seed: Optional[int] = None,
) -> tuple[Optional[int], OptimizationResult, QuantumSolveDetail]:
    settings = get_settings()
    problem = build_problem_for_request(session, tickers, universe_size)
    result, detail = optimize_quantum(
        problem,
        constraints,
        backend=backend,
        target_cardinality=target_cardinality,
        reps=reps,
        shots=shots,
        seed=seed if seed is not None else settings.random_seed,
        risk_free_rate=settings.risk_free_rate,
    )
    run_id: Optional[int] = None
    if persist:
        constraint_id = _persist_constraint(session, constraints)
        run = persist_result(
            session,
            result,
            experiment_id=experiment_id,
            constraint_id=constraint_id,
            seed=seed,
            quantum_detail=detail,
        )
        run_id = run.run_id
    return run_id, result, detail
