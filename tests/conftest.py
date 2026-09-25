"""Shared pytest fixtures.

Integration fixtures use a throwaway SQLite database so the suite needs no
external services. Unit fixtures build domain objects directly from the
deterministic synthetic generator (no DB).
"""

from __future__ import annotations

import pytest

from quantumrisklab.data.cleaning import clean_prices, compute_returns
from quantumrisklab.data.synthetic import generate_universe
from quantumrisklab.domain import PortfolioConstraints, PortfolioProblem
from quantumrisklab.features.problem import build_problem_from_frame


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """A migrated, empty SQLite database wired through the app config."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("QRL_DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("QRL_N_ASSETS", "10")
    monkeypatch.setenv("QRL_N_TRADING_DAYS", "260")

    from quantumrisklab.config import get_settings
    from quantumrisklab.db.connection import get_engine, reset_engine
    from quantumrisklab.db.migrate import run_migrations

    get_settings.cache_clear()
    reset_engine()
    engine = get_engine()
    run_migrations(engine)

    yield engine

    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def seeded_db(tmp_db):
    """A SQLite database seeded with a small synthetic universe."""
    from quantumrisklab.data.pipeline import seed_market_data

    seed_market_data(force=True)
    return tmp_db


@pytest.fixture
def sample_problem() -> PortfolioProblem:
    """A 10-asset PortfolioProblem built without touching the database."""
    universe = generate_universe(n_assets=10, n_trading_days=260, seed=42)
    prices = clean_prices(universe.prices)
    simple, _ = compute_returns(prices)
    sectors = dict(zip(universe.tickers, universe.sectors))
    return build_problem_from_frame(simple, sectors)


@pytest.fixture
def base_constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        name="test",
        max_holdings=4,
        min_weight=0.05,
        max_weight=0.40,
        max_sector_exposure=0.60,
        transaction_cost=0.001,
        risk_aversion=5.0,
    )
