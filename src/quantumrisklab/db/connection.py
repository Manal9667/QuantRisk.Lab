"""Engine and session management.

A single lazily-created engine is shared per process. ``session_scope`` provides
a transactional context manager; ``get_session`` yields a session for FastAPI
dependency injection.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantumrisklab.config import Settings, get_settings

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker[Session]] = None


def get_engine(settings: Optional[Settings] = None) -> Engine:
    """Return the shared SQLAlchemy engine, creating it on first use."""
    global _engine, _SessionFactory
    if _engine is None:
        settings = settings or get_settings()
        connect_args = {}
        if settings.is_sqlite:
            # Allow cross-thread use for the test/dev SQLite fallback.
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            settings.sqlalchemy_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def _get_factory() -> sessionmaker[Session]:
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None  # for type-checkers
    return _SessionFactory


def reset_engine() -> None:
    """Dispose and clear the cached engine (used by tests switching databases)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    session = _get_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a session and always closes it (read-only)."""
    session = _get_factory()()
    try:
        yield session
    finally:
        session.close()


def get_db_session() -> Iterator[Session]:
    """FastAPI dependency that commits on success and rolls back on error."""
    session = _get_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
