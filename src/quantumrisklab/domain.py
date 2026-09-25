"""Core domain dataclasses shared across the feature, optimization, and quantum
layers. Defined in one place to keep the layers decoupled and avoid circular
imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PortfolioConstraints:
    """A constraint / objective specification for portfolio optimization.

    The objective optimized is:
        maximize   expected_return - risk_aversion * variance - transaction_cost
    subject to the constraints below.
    """

    name: str = "default"
    max_holdings: Optional[int] = None          # cardinality cap K
    min_weight: float = 0.0                      # min weight of a *held* position
    max_weight: float = 1.0                      # max weight of any position
    max_volatility: Optional[float] = None       # annualised volatility cap
    max_sector_exposure: Optional[float] = None  # per-sector weight cap
    transaction_cost: float = 0.0                # proportional cost per unit turnover
    risk_aversion: float = 5.0                   # lambda in the objective
    target_beta: Optional[float] = None          # optional beta target
    beta_tolerance: Optional[float] = None       # allowed +/- band around target


@dataclass
class PortfolioProblem:
    """A fully specified optimization instance derived from market data.

    All statistics are precomputed so optimizers are pure functions of this
    object plus a ``PortfolioConstraints``.
    """

    tickers: list[str]
    sectors: list[str]
    expected_returns: np.ndarray   # annualised mean returns, shape (n,)
    cov_matrix: np.ndarray         # annualised covariance, shape (n, n)
    betas: np.ndarray              # per-asset beta vs benchmark, shape (n,)
    daily_mean: np.ndarray         # daily mean returns, shape (n,)
    daily_cov: np.ndarray          # daily covariance, shape (n, n)
    returns_matrix: np.ndarray     # daily returns, shape (T, n)
    market_returns: np.ndarray     # benchmark daily returns, shape (T,)
    prev_weights: Optional[np.ndarray] = None  # for turnover / tx cost, shape (n,)
    trading_days: int = 252

    @property
    def n(self) -> int:
        return len(self.tickers)

    def __post_init__(self) -> None:
        n = len(self.tickers)
        if self.expected_returns.shape != (n,):
            raise ValueError("expected_returns shape mismatch")
        if self.cov_matrix.shape != (n, n):
            raise ValueError("cov_matrix shape mismatch")
        if len(self.sectors) != n:
            raise ValueError("sectors length mismatch")
        if self.betas.shape != (n,):
            raise ValueError("betas shape mismatch")


@dataclass
class PortfolioMetrics:
    """Financial risk/return snapshot of a portfolio."""

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
    sector_exposure: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, float]:
        """Flat mapping suitable for persistence into ``risk_metrics``."""
        return {
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "beta": self.beta,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "turnover": self.turnover,
            "transaction_cost_paid": self.transaction_cost_paid,
            "diversification_ratio": self.diversification_ratio,
            "effective_n": self.effective_n,
            "max_sector_exposure": self.max_sector_exposure,
            "constraint_violation": self.constraint_violation,
        }


@dataclass
class OptimizationResult:
    """Result of a classical or quantum optimization run."""

    run_type: str            # 'classical' | 'quantum'
    solver: str
    tickers: list[str]
    weights: np.ndarray      # shape (n,), sums to ~1 for a funded portfolio
    selected: np.ndarray     # boolean mask, shape (n,)
    objective_value: float
    runtime_seconds: float
    universe_size: int
    n_variables: int
    metrics: Optional[PortfolioMetrics] = None
    status: str = "completed"
    notes: Optional[str] = None

    @property
    def n_selected(self) -> int:
        return int(np.count_nonzero(self.selected))


@dataclass
class QuboArtifacts:
    """The QUBO produced from a cardinality-constrained selection problem.

    Encodes ``x^T Q x + offset`` over binary selection variables ``x``.
    """

    Q: np.ndarray            # symmetric QUBO matrix, shape (n, n)
    offset: float
    n_variables: int
    penalty_cardinality: float
    penalty_diversification: float
    construction_time_seconds: float
    target_cardinality: int

    def energy(self, x: np.ndarray) -> float:
        """QUBO objective for a binary vector ``x`` (lower is better)."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)


@dataclass
class QuantumSolveDetail:
    """Quantum-specific execution detail attached to a quantum run."""

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
    solution_quality: Optional[float] = None
