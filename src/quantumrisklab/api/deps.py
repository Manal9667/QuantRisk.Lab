"""FastAPI dependencies and response builders."""

from __future__ import annotations

import numpy as np

from quantumrisklab.api.schemas import (
    MetricsResponse,
    QuantumDetailResponse,
    RunResponse,
    WeightItem,
)
from quantumrisklab.domain import OptimizationResult, QuantumSolveDetail


def result_to_response(
    result: OptimizationResult,
    run_id: int | None,
    detail: QuantumSolveDetail | None = None,
    include_zero_weights: bool = False,
) -> RunResponse:
    """Convert domain objects to the API response schema."""
    weights: list[WeightItem] = []
    for ticker, w, sel in zip(result.tickers, result.weights, result.selected):
        if include_zero_weights or bool(sel) or abs(float(w)) > 1e-9:
            weights.append(WeightItem(ticker=ticker, weight=float(w), selected=bool(sel)))
    weights.sort(key=lambda x: x.weight, reverse=True)

    metrics = None
    if result.metrics is not None:
        m = result.metrics
        metrics = MetricsResponse(
            expected_return=m.expected_return,
            volatility=m.volatility,
            sharpe_ratio=m.sharpe_ratio,
            max_drawdown=m.max_drawdown,
            beta=m.beta,
            var_95=m.var_95,
            cvar_95=m.cvar_95,
            turnover=m.turnover,
            transaction_cost_paid=m.transaction_cost_paid,
            diversification_ratio=m.diversification_ratio,
            effective_n=m.effective_n,
            max_sector_exposure=m.max_sector_exposure,
            constraint_violation=m.constraint_violation,
            sector_exposure=m.sector_exposure,
        )

    quantum_detail = None
    if detail is not None:
        quantum_detail = QuantumDetailResponse(
            backend=detail.backend,
            n_qubits=detail.n_qubits,
            n_variables=detail.n_variables,
            qubo_construction_time=detail.qubo_construction_time,
            quantum_runtime=detail.quantum_runtime,
            penalty_cardinality=detail.penalty_cardinality,
            penalty_diversification=detail.penalty_diversification,
            reps=detail.reps,
            shots=detail.shots,
            best_bitstring=detail.best_bitstring,
            objective_qubo=detail.objective_qubo,
            num_feasible_samples=detail.num_feasible_samples,
            solution_quality=detail.solution_quality,
        )

    return RunResponse(
        run_id=run_id,
        run_type=result.run_type,
        solver=result.solver,
        universe_size=result.universe_size,
        n_variables=result.n_variables,
        n_selected=int(np.count_nonzero(result.selected)),
        objective_value=result.objective_value,
        runtime_seconds=result.runtime_seconds,
        weights=weights,
        metrics=metrics,
        quantum_detail=quantum_detail,
        notes=result.notes,
    )
