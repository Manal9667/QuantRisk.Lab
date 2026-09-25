"""SQLAlchemy 2.0 ORM models mirroring the PostgreSQL schema (migration 001).

These models are the authoritative Python view of the schema. For PostgreSQL the
raw SQL migration is applied; for SQLite (tests / zero-infra local runs) the
tables are created from this metadata. The two are kept deliberately aligned.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# BIGINT on PostgreSQL, but INTEGER on SQLite so the primary key aliases rowid
# and auto-increments (SQLite only auto-assigns INTEGER PRIMARY KEY).
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


class Asset(Base):
    __tablename__ = "assets"

    asset_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    sector: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False, default="equity")
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    prices: Mapped[list["MarketPrice"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    returns: Mapped[list["DailyReturn"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class MarketPrice(Base):
    __tablename__ = "market_prices"
    __table_args__ = (UniqueConstraint("asset_id", "price_date", name="uq_market_prices"),)

    price_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False
    )
    price_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)

    asset: Mapped["Asset"] = relationship(back_populates="prices")


class DailyReturn(Base):
    __tablename__ = "daily_returns"
    __table_args__ = (UniqueConstraint("asset_id", "return_date", name="uq_daily_returns"),)

    return_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False
    )
    return_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    simple_return: Mapped[float] = mapped_column(Float, nullable=False)
    log_return: Mapped[float] = mapped_column(Float, nullable=False)

    asset: Mapped["Asset"] = relationship(back_populates="returns")


class PortfolioConstraint(Base):
    __tablename__ = "portfolio_constraints"

    constraint_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    max_holdings: Mapped[Optional[int]] = mapped_column(Integer)
    min_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    max_volatility: Mapped[Optional[float]] = mapped_column(Float)
    max_sector_exposure: Mapped[Optional[float]] = mapped_column(Float)
    transaction_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_aversion: Mapped[float] = mapped_column(Float, nullable=False, default=5.0)
    target_beta: Mapped[Optional[float]] = mapped_column(Float)
    beta_tolerance: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    runs: Mapped[list["OptimizationRun"]] = relationship(back_populates="experiment")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    __table_args__ = (
        CheckConstraint("run_type IN ('classical', 'quantum')", name="ck_run_type"),
    )

    run_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("experiments.experiment_id", ondelete="SET NULL")
    )
    constraint_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("portfolio_constraints.constraint_id", ondelete="SET NULL")
    )
    run_type: Mapped[str] = mapped_column(String(16), nullable=False)
    solver: Mapped[str] = mapped_column(String(64), nullable=False)
    universe_size: Mapped[int] = mapped_column(Integer, nullable=False)
    n_variables: Mapped[int] = mapped_column(Integer, nullable=False)
    n_selected: Mapped[Optional[int]] = mapped_column(Integer)
    objective_value: Mapped[Optional[float]] = mapped_column(Float)
    runtime_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="completed")
    seed: Mapped[Optional[int]] = mapped_column(Integer)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    experiment: Mapped[Optional["Experiment"]] = relationship(back_populates="runs")
    constraint: Mapped[Optional["PortfolioConstraint"]] = relationship()
    weights: Mapped[list["PortfolioWeight"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    risk: Mapped[Optional["RiskMetric"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )
    quantum: Mapped[Optional["QuantumRun"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )


class PortfolioWeight(Base):
    __tablename__ = "portfolio_weights"
    __table_args__ = (UniqueConstraint("run_id", "asset_id", name="uq_portfolio_weights"),)

    weight_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("optimization_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False
    )
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    run: Mapped["OptimizationRun"] = relationship(back_populates="weights")
    asset: Mapped["Asset"] = relationship()


class RiskMetric(Base):
    __tablename__ = "risk_metrics"

    metric_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("optimization_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    expected_return: Mapped[Optional[float]] = mapped_column(Float)
    volatility: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    beta: Mapped[Optional[float]] = mapped_column(Float)
    var_95: Mapped[Optional[float]] = mapped_column(Float)
    cvar_95: Mapped[Optional[float]] = mapped_column(Float)
    turnover: Mapped[Optional[float]] = mapped_column(Float)
    transaction_cost_paid: Mapped[Optional[float]] = mapped_column(Float)
    diversification_ratio: Mapped[Optional[float]] = mapped_column(Float)
    effective_n: Mapped[Optional[float]] = mapped_column(Float)
    max_sector_exposure: Mapped[Optional[float]] = mapped_column(Float)
    constraint_violation: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run: Mapped["OptimizationRun"] = relationship(back_populates="risk")


class QuantumRun(Base):
    __tablename__ = "quantum_runs"

    quantum_run_id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("optimization_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    n_qubits: Mapped[int] = mapped_column(Integer, nullable=False)
    n_variables: Mapped[int] = mapped_column(Integer, nullable=False)
    qubo_construction_time: Mapped[float] = mapped_column(Float, nullable=False)
    quantum_runtime: Mapped[float] = mapped_column(Float, nullable=False)
    penalty_cardinality: Mapped[Optional[float]] = mapped_column(Float)
    penalty_diversification: Mapped[Optional[float]] = mapped_column(Float)
    reps: Mapped[Optional[int]] = mapped_column(Integer)
    shots: Mapped[Optional[int]] = mapped_column(Integer)
    best_bitstring: Mapped[Optional[str]] = mapped_column(String(256))
    objective_qubo: Mapped[Optional[float]] = mapped_column(Float)
    num_feasible_samples: Mapped[Optional[int]] = mapped_column(Integer)
    solution_quality: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run: Mapped["OptimizationRun"] = relationship(back_populates="quantum")
