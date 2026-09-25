"""Analytical SQL query library.

Loads the ``.sql`` files under ``sql/queries`` and executes them against the
active database, returning plain ``list[dict]`` rows. Keeping the analytics in
SQL (window functions, conditional aggregates, joins across the run tables) is a
deliberate architectural choice: the database does real analytical work, not
just storage.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Engine, text
from sqlalchemy.engine import Row

from quantumrisklab.config import get_settings

# Repo root: src/quantumrisklab/db/queries.py -> parents[3] == repo root.
_QUERY_DIR = Path(__file__).resolve().parents[3] / "sql" / "queries"


# Strip ``-- ...`` line comments. Necessary because SQLAlchemy's text() treats
# ``:name`` tokens as bind parameters even inside comments, and our comments
# document the parameter names using that syntax.
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


@lru_cache
def _load_sql(name: str) -> str:
    path = _QUERY_DIR / f"{name}.sql"
    if not path.exists():
        raise FileNotFoundError(f"SQL query file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    return _LINE_COMMENT_RE.sub("", raw)


def _rows_to_dicts(rows: list[Row[Any]]) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in rows]


def _execute(engine: Engine, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return _rows_to_dicts(result.fetchall())


class QueryLibrary:
    """Runs the named analytical queries against a given engine."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # -- calculating historical returns ------------------------------------
    def historical_returns(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        sql = _load_sql("historical_returns")
        return _execute(
            self.engine,
            sql,
            {"ticker": ticker, "start_date": start_date, "end_date": end_date},
        )

    # -- retrieving rolling volatility -------------------------------------
    def rolling_volatility(
        self,
        window: int = 21,
        ticker: Optional[str] = None,
        trading_days_per_year: int = 252,
    ) -> list[dict[str, Any]]:
        if window < 2:
            raise ValueError("window must be >= 2 for a sample standard deviation")
        # window is validated as an int here; it is safe to format into the SQL
        # frame clause (bind parameters are not permitted in frame offsets).
        sql = _load_sql("rolling_volatility").format(
            window=int(window),
            window_minus_1=int(window) - 1,
            ann=int(trading_days_per_year),
        )
        return _execute(self.engine, sql, {"ticker": ticker})

    # -- comparing portfolio risk across optimization runs -----------------
    def compare_risk_across_runs(
        self, experiment_id: Optional[int] = None
    ) -> list[dict[str, Any]]:
        sql = _load_sql("compare_risk_across_runs")
        return _execute(self.engine, sql, {"experiment_id": experiment_id})

    # -- finding the assets selected most frequently -----------------------
    def most_selected_assets(
        self, run_type: Optional[str] = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        sql = _load_sql("most_selected_assets")
        return _execute(self.engine, sql, {"run_type": run_type, "limit": int(limit)})

    # -- comparing classical vs quantum solutions --------------------------
    def classical_vs_quantum(
        self, experiment_id: Optional[int] = None
    ) -> list[dict[str, Any]]:
        sql = _load_sql("classical_vs_quantum")
        return _execute(self.engine, sql, {"experiment_id": experiment_id})

    # -- tracking optimization performance over time -----------------------
    def optimization_performance_over_time(
        self, experiment_id: Optional[int] = None
    ) -> list[dict[str, Any]]:
        sql = _load_sql("optimization_performance_over_time")
        return _execute(self.engine, sql, {"experiment_id": experiment_id})


def get_query_library(engine: Optional[Engine] = None) -> QueryLibrary:
    """Convenience factory using the shared engine when none is supplied."""
    if engine is None:
        from quantumrisklab.db.connection import get_engine

        engine = get_engine(get_settings())
    return QueryLibrary(engine)
