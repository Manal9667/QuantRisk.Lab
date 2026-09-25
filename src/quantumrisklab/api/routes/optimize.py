"""Optimization endpoints: POST /optimize/classical and POST /optimize/quantum."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from quantumrisklab.api.deps import result_to_response
from quantumrisklab.api.schemas import OptimizeRequest, QuantumOptimizeRequest, RunResponse
from quantumrisklab.db.connection import get_db_session
from quantumrisklab.service import run_classical, run_quantum

router = APIRouter(prefix="/optimize", tags=["optimize"])


@router.post("/classical", response_model=RunResponse)
def optimize_classical_endpoint(
    request: OptimizeRequest,
    session: Session = Depends(get_db_session),
) -> RunResponse:
    try:
        run_id, result = run_classical(
            session,
            constraints=request.constraints.to_domain(),
            tickers=request.tickers,
            universe_size=request.universe_size,
            experiment_id=request.experiment_id,
            persist=request.persist,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result_to_response(result, run_id)


@router.post("/quantum", response_model=RunResponse)
def optimize_quantum_endpoint(
    request: QuantumOptimizeRequest,
    session: Session = Depends(get_db_session),
) -> RunResponse:
    try:
        run_id, result, detail = run_quantum(
            session,
            constraints=request.constraints.to_domain(),
            backend=request.backend,
            target_cardinality=request.target_cardinality,
            reps=request.reps,
            shots=request.shots,
            tickers=request.tickers,
            universe_size=request.universe_size,
            experiment_id=request.experiment_id,
            persist=request.persist,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # e.g. QAOA above the simulator qubit cap, or Qiskit not installed.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result_to_response(result, run_id, detail=detail)
