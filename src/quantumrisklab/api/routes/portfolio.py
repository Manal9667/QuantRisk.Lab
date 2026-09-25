"""Portfolio / risk retrieval: GET /portfolio/{id} and GET /risk/{portfolio_id}.

Here a "portfolio" is identified by its optimization run id.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from quantumrisklab.db.connection import get_session
from quantumrisklab.db.repository import get_risk_metric, get_run, get_weights

router = APIRouter(tags=["portfolio"])


@router.get("/portfolio/{run_id}")
def get_portfolio(run_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    run = get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    weights = get_weights(session, run_id)
    return {
        "run_id": run.run_id,
        "run_type": run.run_type,
        "solver": run.solver,
        "experiment_id": run.experiment_id,
        "universe_size": run.universe_size,
        "n_variables": run.n_variables,
        "n_selected": run.n_selected,
        "objective_value": run.objective_value,
        "runtime_seconds": run.runtime_seconds,
        "created_at": str(run.created_at),
        "weights": [
            {"ticker": t, "weight": w, "selected": s}
            for t, w, s in weights
            if s or abs(w) > 1e-9
        ],
    }


@router.get("/risk/{run_id}")
def get_risk(run_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    run = get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    rm = get_risk_metric(session, run_id)
    if rm is None:
        raise HTTPException(status_code=404, detail=f"no risk metrics for run {run_id}")
    return {
        "run_id": run_id,
        "run_type": run.run_type,
        "expected_return": rm.expected_return,
        "volatility": rm.volatility,
        "sharpe_ratio": rm.sharpe_ratio,
        "max_drawdown": rm.max_drawdown,
        "beta": rm.beta,
        "var_95": rm.var_95,
        "cvar_95": rm.cvar_95,
        "turnover": rm.turnover,
        "transaction_cost_paid": rm.transaction_cost_paid,
        "diversification_ratio": rm.diversification_ratio,
        "effective_n": rm.effective_n,
        "max_sector_exposure": rm.max_sector_exposure,
        "constraint_violation": rm.constraint_violation,
    }
