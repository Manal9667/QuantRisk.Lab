"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from quantumrisklab.domain import PortfolioConstraints


class ConstraintSpec(BaseModel):
    """User-supplied constraint / objective specification."""

    name: str = "api-request"
    max_holdings: Optional[int] = Field(default=None, ge=1)
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    max_volatility: Optional[float] = Field(default=None, gt=0.0)
    max_sector_exposure: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    transaction_cost: float = Field(default=0.0, ge=0.0)
    risk_aversion: float = Field(default=5.0, ge=0.0)
    target_beta: Optional[float] = None
    beta_tolerance: Optional[float] = Field(default=None, ge=0.0)

    def to_domain(self) -> PortfolioConstraints:
        return PortfolioConstraints(
            name=self.name,
            max_holdings=self.max_holdings,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            max_volatility=self.max_volatility,
            max_sector_exposure=self.max_sector_exposure,
            transaction_cost=self.transaction_cost,
            risk_aversion=self.risk_aversion,
            target_beta=self.target_beta,
            beta_tolerance=self.beta_tolerance,
        )


class OptimizeRequest(BaseModel):
    """Common fields for optimization requests."""

    tickers: Optional[list[str]] = Field(
        default=None, description="Explicit universe. Overrides universe_size."
    )
    universe_size: Optional[int] = Field(
        default=None, ge=1, description="Use the first N stored assets."
    )
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    persist: bool = True
    experiment_id: Optional[int] = None
    seed: Optional[int] = None


class QuantumOptimizeRequest(OptimizeRequest):
    backend: Literal["exact", "qaoa", "numpy_min_eigen"] = "exact"
    target_cardinality: Optional[int] = Field(default=None, ge=1)
    reps: int = Field(default=1, ge=1)
    shots: int = Field(default=1024, ge=1)


class WeightItem(BaseModel):
    ticker: str
    weight: float
    selected: bool


class MetricsResponse(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    var_95: float
    cvar_95: float
    turnover: float
    transaction_cost_paid: float
    diversification_ratio: float
    effective_n: float
    max_sector_exposure: float
    constraint_violation: float
    sector_exposure: dict[str, float] = {}


class QuantumDetailResponse(BaseModel):
    backend: str
    n_qubits: int
    n_variables: int
    qubo_construction_time: float
    quantum_runtime: float
    penalty_cardinality: float
    penalty_diversification: float
    reps: Optional[int]
    shots: Optional[int]
    best_bitstring: str
    objective_qubo: float
    num_feasible_samples: int
    solution_quality: Optional[float]


class RunResponse(BaseModel):
    run_id: Optional[int]
    run_type: str
    solver: str
    universe_size: int
    n_variables: int
    n_selected: int
    objective_value: float
    runtime_seconds: float
    weights: list[WeightItem]
    metrics: Optional[MetricsResponse]
    quantum_detail: Optional[QuantumDetailResponse] = None
    notes: Optional[str] = None


class AssetResponse(BaseModel):
    asset_id: int
    ticker: str
    name: str
    sector: str
    asset_class: str
    currency: str


class ExperimentResponse(BaseModel):
    experiment_id: int
    name: str
    seed: int
    description: Optional[str]
    created_at: str
