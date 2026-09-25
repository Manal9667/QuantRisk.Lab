"""Lightweight, transparent SQL migration runner.

Applies the versioned ``.sql`` files in the top-level ``migrations`` directory in
order, tracking which have been applied in a ``schema_migrations`` table. This is
intentionally simpler than a full framework: the migrations are plain SQL you can
read, and application is idempotent.

For SQLite (tests / zero-infra dev), the raw PostgreSQL DDL is not used; instead
``create_all_sqlite`` builds the schema from the ORM metadata.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import Engine, text

from quantumrisklab.config import get_settings
from quantumrisklab.db.connection import get_engine
from quantumrisklab.db.models import Base

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

# Naive statement splitter: our migrations use plain statements terminated by ';'
# with no PL/pgSQL bodies, so splitting on semicolons (ignoring comments) is safe.
_COMMENT_RE = re.compile(r"^\s*--.*$", re.MULTILINE)


def _split_statements(sql: str) -> list[str]:
    stripped = _COMMENT_RE.sub("", sql)
    return [s.strip() for s in stripped.split(";") if s.strip()]


def _ensure_version_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version VARCHAR(64) PRIMARY KEY,"
                " applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )


def _applied_versions(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT version FROM schema_migrations")).fetchall()
    return {r[0] for r in rows}


def discover_migrations() -> list[Path]:
    if not _MIGRATIONS_DIR.exists():
        return []
    return sorted(_MIGRATIONS_DIR.glob("*.sql"))


def create_all_sqlite(engine: Engine) -> None:
    """Create the full schema from ORM metadata (used for SQLite)."""
    Base.metadata.create_all(engine)


def run_migrations(engine: Optional[Engine] = None) -> list[str]:
    """Apply pending migrations. Returns the list of versions applied this call.

    On SQLite the ORM metadata is used directly (the raw DDL is PostgreSQL
    specific); on other backends the SQL migration files are applied in order.
    """
    settings = get_settings()
    engine = engine or get_engine(settings)

    if settings.is_sqlite:
        create_all_sqlite(engine)
        logger.info("SQLite schema created from ORM metadata.")
        return ["orm_metadata"]

    _ensure_version_table(engine)
    already = _applied_versions(engine)
    applied_now: list[str] = []

    for path in discover_migrations():
        version = path.stem
        if version in already:
            continue
        logger.info("Applying migration %s", version)
        statements = _split_statements(path.read_text(encoding="utf-8"))
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
            conn.execute(
                text("INSERT INTO schema_migrations (version) VALUES (:v)"),
                {"v": version},
            )
        applied_now.append(version)

    if not applied_now:
        logger.info("No pending migrations.")
    return applied_now
