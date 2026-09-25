"""Centralised, environment-driven configuration.

Every tunable parameter is surfaced here so experiments are reproducible and the
system can be reconfigured entirely through environment variables (or a ``.env``
file). Nothing about the financial results is hardcoded downstream; defaults live
here and can be overridden per-run.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

QuantumBackend = Literal["exact", "qaoa", "numpy_min_eigen"]


class Settings(BaseSettings):
    """Application settings, populated from environment variables.

    Environment variables are prefixed with ``QRL_`` (e.g. ``QRL_DB_HOST``).
    """

    model_config = SettingsConfigDict(
        env_prefix="QRL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---------------------------------------------------------
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "quantumrisklab"
    db_user: str = "qrl"
    db_password: str = "qrl_password"
    database_url: Optional[str] = Field(
        default=None,
        description="Full SQLAlchemy URL. Overrides discrete db_* fields when set.",
    )

    # --- Data pipeline ----------------------------------------------------
    random_seed: int = 42
    n_assets: int = 60
    n_trading_days: int = 756  # ~3 years of trading days
    risk_free_rate: float = 0.02  # annualised

    # --- Optimization defaults -------------------------------------------
    risk_aversion: float = 5.0
    transaction_cost: float = 0.001  # proportional, per unit turnover

    # --- Quantum backend --------------------------------------------------
    quantum_backend: QuantumBackend = "exact"
    qaoa_reps: int = 1
    qaoa_shots: int = 1024

    # --- API --------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        """Resolve the effective SQLAlchemy connection URL.

        Precedence: explicit ``database_url`` > discrete PostgreSQL fields.
        The PostgreSQL URL uses the psycopg (v3) driver.
        """
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def is_sqlite(self) -> bool:
        return self.sqlalchemy_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Cached so the same configuration object is shared across the process. Tests
    that need to override configuration can call ``get_settings.cache_clear()``.
    """
    return Settings()
