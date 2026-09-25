"""Database initialisation entry point.

Runs migrations and (optionally) seeds the investable universe with deterministic
synthetic market data. Exposed as the ``quantumrisklab-initdb`` console script.

Usage:
    python -m quantumrisklab.db.init_db [--seed-data] [--force]
"""

from __future__ import annotations

import argparse
import logging

from quantumrisklab.config import get_settings
from quantumrisklab.db.connection import get_engine
from quantumrisklab.db.migrate import run_migrations

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialise the QuantumRiskLab database.")
    parser.add_argument(
        "--seed-data",
        action="store_true",
        help="Generate and load deterministic synthetic market data after migrating.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-seed data even if assets already exist.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=get_settings().log_level)

    engine = get_engine()
    applied = run_migrations(engine)
    logger.info("Migrations applied: %s", applied or "(none pending)")

    if args.seed_data:
        # Imported lazily to keep the migration path free of heavy imports.
        from quantumrisklab.data.pipeline import seed_market_data

        n = seed_market_data(force=args.force)
        logger.info("Seeded %d asset price series.", n)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
