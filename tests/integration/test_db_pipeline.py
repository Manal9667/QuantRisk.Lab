"""Integration tests for the data pipeline, repository, and analytical SQL."""

from __future__ import annotations

import pytest

from quantumrisklab.db.connection import session_scope
from quantumrisklab.db.queries import QueryLibrary
from quantumrisklab.db.repository import count_assets, list_assets, load_returns_frame

pytestmark = pytest.mark.integration


def test_seed_populates_assets_and_returns(seeded_db):
    with session_scope() as session:
        assert count_assets(session) == 10
        assets = list_assets(session)
        assert len(assets) == 10
        returns = load_returns_frame(session)
        assert returns.shape[1] == 10
        assert not returns.empty


def test_seed_is_idempotent(seeded_db):
    from quantumrisklab.data.pipeline import seed_market_data

    # Without force, a second seed is a no-op.
    added = seed_market_data(force=False)
    assert added == 0
    with session_scope() as session:
        assert count_assets(session) == 10


def test_historical_returns_query(seeded_db):
    ql = QueryLibrary(seeded_db)
    rows = ql.historical_returns(ticker="AST000")
    assert len(rows) > 0
    # First row has no prior price so simple_return is NULL; later rows populated.
    populated = [r for r in rows if r["simple_return"] is not None]
    assert len(populated) > 0


def test_rolling_volatility_query(seeded_db):
    ql = QueryLibrary(seeded_db)
    rows = ql.rolling_volatility(window=21, ticker="AST000")
    vols = [r["rolling_vol_annualized"] for r in rows if r["rolling_vol_annualized"] is not None]
    assert len(vols) > 0
    assert all(v >= 0 for v in vols)


def test_analytics_queries_run_when_empty(seeded_db):
    # No runs yet -> aggregate queries return empty, not errors.
    ql = QueryLibrary(seeded_db)
    assert ql.most_selected_assets(limit=5) == []
    assert ql.classical_vs_quantum() == []
    assert ql.compare_risk_across_runs() == []
    assert ql.optimization_performance_over_time() == []
