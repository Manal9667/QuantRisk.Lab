"""Repository: persistence and retrieval helpers built on the ORM.

Keeps SQLAlchemy specifics out of the optimization/quantum layers. Optimization
results (domain dataclasses) go in; typed rows and pandas frames come out.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from quantumrisklab.domain import (
    OptimizationResult,
    PortfolioConstraints,
    QuantumSolveDetail,
)
from quantumrisklab.db.models import (
    Asset,
    DailyReturn,
    Experiment,
    MarketPrice,
    OptimizationRun,
    PortfolioConstraint,
    PortfolioWeight,
    QuantumRun,
    RiskMetric,
)


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
def list_assets(session: Session) -> list[Asset]:
    return list(session.scalars(select(Asset).order_by(Asset.ticker)))


def get_asset_by_ticker(session: Session, ticker: str) -> Optional[Asset]:
    return session.scalar(select(Asset).where(Asset.ticker == ticker))


def count_assets(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(Asset)) or 0)


# --------------------------------------------------------------------------- #
# Returns / prices as matrices
# --------------------------------------------------------------------------- #
def load_returns_frame(
    session: Session, tickers: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Return a (dates x tickers) DataFrame of simple daily returns.

    Only dates present for *all* requested tickers are kept, so the resulting
    matrix is rectangular and ready for covariance estimation.
    """
    stmt = (
        select(Asset.ticker, DailyReturn.return_date, DailyReturn.simple_return)
        .join(DailyReturn, DailyReturn.asset_id == Asset.asset_id)
        .order_by(DailyReturn.return_date)
    )
    if tickers is not None:
        stmt = stmt.where(Asset.ticker.in_(list(tickers)))
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ticker", "return_date", "simple_return"])
    wide = df.pivot(index="return_date", columns="ticker", values="simple_return")
    wide = wide.dropna(how="any")
    return wide


def load_prices_frame(
    session: Session, tickers: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    stmt = (
        select(Asset.ticker, MarketPrice.price_date, MarketPrice.adj_close)
        .join(MarketPrice, MarketPrice.asset_id == Asset.asset_id)
        .order_by(MarketPrice.price_date)
    )
    if tickers is not None:
        stmt = stmt.where(Asset.ticker.in_(list(tickers)))
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ticker", "price_date", "adj_close"])
    return df.pivot(index="price_date", columns="ticker", values="adj_close")


def sector_map(session: Session, tickers: Optional[Sequence[str]] = None) -> dict[str, str]:
    stmt = select(Asset.ticker, Asset.sector)
    if tickers is not None:
        stmt = stmt.where(Asset.ticker.in_(list(tickers)))
    return {t: s for t, s in session.execute(stmt).all()}


# --------------------------------------------------------------------------- #
# Constraints / experiments
# --------------------------------------------------------------------------- #
def create_constraint(session: Session, c: PortfolioConstraints) -> PortfolioConstraint:
    row = PortfolioConstraint(
        name=c.name,
        max_holdings=c.max_holdings,
        min_weight=c.min_weight,
        max_weight=c.max_weight,
        max_volatility=c.max_volatility,
        max_sector_exposure=c.max_sector_exposure,
        transaction_cost=c.transaction_cost,
        risk_aversion=c.risk_aversion,
        target_beta=c.target_beta,
        beta_tolerance=c.beta_tolerance,
    )
    session.add(row)
    session.flush()
    return row


def create_experiment(
    session: Session, name: str, seed: int, description: Optional[str] = None
) -> Experiment:
    exp = Experiment(name=name, seed=seed, description=description)
    session.add(exp)
    session.flush()
    return exp


def list_experiments(session: Session) -> list[Experiment]:
    return list(session.scalars(select(Experiment).order_by(Experiment.created_at.desc())))


# --------------------------------------------------------------------------- #
# Persisting optimization results
# --------------------------------------------------------------------------- #
def persist_result(
    session: Session,
    result: OptimizationResult,
    *,
    experiment_id: Optional[int] = None,
    constraint_id: Optional[int] = None,
    seed: Optional[int] = None,
    quantum_detail: Optional[QuantumSolveDetail] = None,
) -> OptimizationRun:
    """Persist a run: optimization_runs + portfolio_weights + risk_metrics
    (+ quantum_runs for quantum runs). Returns the created OptimizationRun.
    """
    run = OptimizationRun(
        experiment_id=experiment_id,
        constraint_id=constraint_id,
        run_type=result.run_type,
        solver=result.solver,
        universe_size=result.universe_size,
        n_variables=result.n_variables,
        n_selected=result.n_selected,
        objective_value=result.objective_value,
        runtime_seconds=result.runtime_seconds,
        status=result.status,
        seed=seed,
        notes=result.notes,
    )
    session.add(run)
    session.flush()  # assigns run.run_id

    # Map tickers -> asset_id once.
    ticker_to_id = {
        t: aid
        for t, aid in session.execute(
            select(Asset.ticker, Asset.asset_id).where(Asset.ticker.in_(result.tickers))
        ).all()
    }
    for i, ticker in enumerate(result.tickers):
        asset_id = ticker_to_id.get(ticker)
        if asset_id is None:
            continue
        session.add(
            PortfolioWeight(
                run_id=run.run_id,
                asset_id=asset_id,
                weight=float(result.weights[i]),
                selected=bool(result.selected[i]),
            )
        )

    if result.metrics is not None:
        m = result.metrics
        session.add(
            RiskMetric(
                run_id=run.run_id,
                turnover=m.turnover,
                transaction_cost_paid=m.transaction_cost_paid,
                **{k: v for k, v in m.to_row().items() if k not in {"turnover", "transaction_cost_paid"}},
            )
        )

    if quantum_detail is not None:
        qd = quantum_detail
        session.add(
            QuantumRun(
                run_id=run.run_id,
                backend=qd.backend,
                n_qubits=qd.n_qubits,
                n_variables=qd.n_variables,
                qubo_construction_time=qd.qubo_construction_time,
                quantum_runtime=qd.quantum_runtime,
                penalty_cardinality=qd.penalty_cardinality,
                penalty_diversification=qd.penalty_diversification,
                reps=qd.reps,
                shots=qd.shots,
                best_bitstring=qd.best_bitstring,
                objective_qubo=qd.objective_qubo,
                num_feasible_samples=qd.num_feasible_samples,
                solution_quality=qd.solution_quality,
            )
        )

    session.flush()
    return run


# --------------------------------------------------------------------------- #
# Retrieval for the API
# --------------------------------------------------------------------------- #
def get_run(session: Session, run_id: int) -> Optional[OptimizationRun]:
    return session.get(OptimizationRun, run_id)


def get_weights(session: Session, run_id: int) -> list[tuple[str, float, bool]]:
    stmt = (
        select(Asset.ticker, PortfolioWeight.weight, PortfolioWeight.selected)
        .join(Asset, Asset.asset_id == PortfolioWeight.asset_id)
        .where(PortfolioWeight.run_id == run_id)
        .order_by(PortfolioWeight.weight.desc())
    )
    return [(t, float(w), bool(s)) for t, w, s in session.execute(stmt).all()]


def get_risk_metric(session: Session, run_id: int) -> Optional[RiskMetric]:
    return session.scalar(select(RiskMetric).where(RiskMetric.run_id == run_id))
