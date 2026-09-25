"""Experiment endpoints: GET /experiments and GET /experiments/compare.

These lean on the analytical SQL queries so the database does the aggregation.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from quantumrisklab.api.schemas import ExperimentResponse
from quantumrisklab.db.connection import get_engine, get_session
from quantumrisklab.db.queries import QueryLibrary
from quantumrisklab.db.repository import list_experiments

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def get_experiments(session: Session = Depends(get_session)) -> list[ExperimentResponse]:
    experiments = list_experiments(session)
    return [
        ExperimentResponse(
            experiment_id=e.experiment_id,
            name=e.name,
            seed=e.seed,
            description=e.description,
            created_at=str(e.created_at),
        )
        for e in experiments
    ]


@router.get("/compare")
def compare_experiments(
    experiment_id: Optional[int] = Query(default=None),
) -> dict[str, Any]:
    """Side-by-side classical vs quantum comparison and performance-over-time,
    computed in SQL."""
    ql = QueryLibrary(get_engine())
    return {
        "classical_vs_quantum": ql.classical_vs_quantum(experiment_id),
        "performance_over_time": ql.optimization_performance_over_time(experiment_id),
        "risk_across_runs": ql.compare_risk_across_runs(experiment_id),
    }
