"""Data pipeline: deterministic synthetic generation, cleaning, ingestion, and
persistence into the database."""

from __future__ import annotations

from quantumrisklab.data.pipeline import build_universe, seed_market_data

__all__ = ["build_universe", "seed_market_data"]
