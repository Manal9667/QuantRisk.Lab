"""End-to-end data pipeline: ingest -> clean -> compute returns -> persist.

``build_universe`` runs the in-memory pipeline and returns cleaned frames.
``seed_market_data`` additionally persists assets, prices, and returns into the
database so SQL analytics and the API have data to work with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quantumrisklab.config import Settings, get_settings
from quantumrisklab.data.cleaning import clean_prices, compute_returns
from quantumrisklab.data.ingestion import PriceProvider, SyntheticProvider
from quantumrisklab.db.connection import session_scope
from quantumrisklab.db.models import Asset, DailyReturn, MarketPrice
from quantumrisklab.db.repository import count_assets

logger = logging.getLogger(__name__)


@dataclass
class UniverseData:
    """In-memory result of the data pipeline."""

    tickers: list[str]
    names: list[str]
    sectors: list[str]
    prices: pd.DataFrame          # cleaned adjusted close
    simple_returns: pd.DataFrame  # daily simple returns
    log_returns: pd.DataFrame     # daily log returns


def build_universe(
    provider: Optional[PriceProvider] = None,
    settings: Optional[Settings] = None,
) -> UniverseData:
    """Ingest and clean the universe, returning cleaned frames (no DB writes)."""
    settings = settings or get_settings()
    provider = provider or SyntheticProvider(
        n_assets=settings.n_assets,
        n_trading_days=settings.n_trading_days,
        seed=settings.random_seed,
    )
    universe = provider.fetch()

    prices = clean_prices(universe.prices)
    surviving = list(prices.columns)
    idx = {t: i for i, t in enumerate(universe.tickers)}
    names = [universe.names[idx[t]] for t in surviving]
    sectors = [universe.sectors[idx[t]] for t in surviving]

    simple_returns, log_returns = compute_returns(prices)

    return UniverseData(
        tickers=surviving,
        names=names,
        sectors=sectors,
        prices=prices,
        simple_returns=simple_returns,
        log_returns=log_returns,
    )


def seed_market_data(
    provider: Optional[PriceProvider] = None,
    settings: Optional[Settings] = None,
    force: bool = False,
) -> int:
    """Persist the universe into the database.

    Returns the number of asset price series written. If assets already exist and
    ``force`` is False, this is a no-op (idempotent seeding).
    """
    settings = settings or get_settings()
    data = build_universe(provider=provider, settings=settings)

    with session_scope() as session:
        existing = count_assets(session)
        if existing > 0 and not force:
            logger.info("Assets already present (%d); skipping seed.", existing)
            return 0

        # Build ticker -> Asset rows.
        assets: dict[str, Asset] = {}
        for ticker, name, sector in zip(data.tickers, data.names, data.sectors):
            asset = Asset(ticker=ticker, name=name, sector=sector, asset_class="equity")
            session.add(asset)
            assets[ticker] = asset
        session.flush()  # assign asset_ids

        # Prices.
        price_records: list[MarketPrice] = []
        for ticker in data.tickers:
            asset_id = assets[ticker].asset_id
            series = data.prices[ticker]
            for date, adj_close in series.items():
                price_records.append(
                    MarketPrice(
                        asset_id=asset_id,
                        price_date=pd.Timestamp(date).date(),
                        open=float(adj_close),
                        high=float(adj_close),
                        low=float(adj_close),
                        close=float(adj_close),
                        adj_close=float(adj_close),
                        volume=None,
                    )
                )
        session.add_all(price_records)

        # Returns.
        return_records: list[DailyReturn] = []
        for ticker in data.tickers:
            asset_id = assets[ticker].asset_id
            simple = data.simple_returns[ticker]
            log = data.log_returns[ticker]
            for date in simple.index:
                return_records.append(
                    DailyReturn(
                        asset_id=asset_id,
                        return_date=pd.Timestamp(date).date(),
                        simple_return=float(simple.loc[date]),
                        log_return=float(log.loc[date]),
                    )
                )
        session.add_all(return_records)

    logger.info(
        "Seeded %d assets, %d price rows, %d return rows.",
        len(data.tickers),
        len(data.prices) * len(data.tickers),
        len(data.simple_returns) * len(data.tickers),
    )
    return len(data.tickers)
