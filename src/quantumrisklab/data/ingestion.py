"""Data ingestion abstraction.

A ``PriceProvider`` supplies raw price history for the universe. The default is
``SyntheticProvider`` (offline, deterministic). The protocol leaves room for a
real market-data provider (e.g. a vendor API) without changing downstream code;
such a provider would live here and implement the same interface.
"""

from __future__ import annotations

from typing import Protocol

from quantumrisklab.data.synthetic import SyntheticUniverse, generate_universe


class PriceProvider(Protocol):
    """Supplies a synthetic-or-real investable universe with price history."""

    def fetch(self) -> SyntheticUniverse:  # pragma: no cover - interface
        ...


class SyntheticProvider:
    """Deterministic synthetic provider driven by a seed."""

    def __init__(self, n_assets: int, n_trading_days: int, seed: int) -> None:
        self.n_assets = n_assets
        self.n_trading_days = n_trading_days
        self.seed = seed

    def fetch(self) -> SyntheticUniverse:
        return generate_universe(
            n_assets=self.n_assets,
            n_trading_days=self.n_trading_days,
            seed=self.seed,
        )
