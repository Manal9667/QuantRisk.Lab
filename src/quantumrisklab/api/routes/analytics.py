"""Analytics endpoints backed by the SQL query library and feature layer.

Provides data for the dashboard: rolling volatility, most-frequently-selected
assets, and the correlation matrix.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from quantumrisklab.db.connection import get_engine, get_session
from quantumrisklab.db.queries import QueryLibrary
from quantumrisklab.db.repository import load_returns_frame

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/rolling-volatility")
def rolling_volatility(
    window: int = Query(default=21, ge=2, le=252),
    ticker: Optional[str] = Query(default=None),
) -> list[dict[str, Any]]:
    ql = QueryLibrary(get_engine())
    return ql.rolling_volatility(window=window, ticker=ticker)


@router.get("/most-selected")
def most_selected(
    run_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict[str, Any]]:
    ql = QueryLibrary(get_engine())
    return ql.most_selected_assets(run_type=run_type, limit=limit)


@router.get("/historical-returns")
def historical_returns(
    ticker: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
) -> list[dict[str, Any]]:
    ql = QueryLibrary(get_engine())
    return ql.historical_returns(ticker=ticker, start_date=start_date, end_date=end_date)


@router.get("/correlation")
def correlation_matrix(
    universe_size: Optional[int] = Query(default=25, ge=2, le=100),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Correlation matrix of daily returns for the first N assets (dashboard heatmap)."""
    returns = load_returns_frame(session)
    if returns.empty:
        return {"tickers": [], "matrix": []}
    if universe_size:
        returns = returns.iloc[:, : int(universe_size)]
    corr = returns.corr()
    return {
        "tickers": list(corr.columns),
        "matrix": corr.to_numpy().round(4).tolist(),
    }
