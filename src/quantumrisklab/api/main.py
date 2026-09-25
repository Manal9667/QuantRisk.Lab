"""FastAPI application factory and app instance.

On startup the app ensures the schema exists (runs migrations). It intentionally
does NOT auto-seed market data; seeding is an explicit operational step
(``init_db --seed-data`` or the experiment runner) so data provenance is clear.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quantumrisklab.api.routes import analytics, assets, experiments, optimize, portfolio
from quantumrisklab.config import get_settings
from quantumrisklab.db.connection import get_engine
from quantumrisklab.db.migrate import run_migrations
from quantumrisklab.quantum.solver import qiskit_available

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    try:
        run_migrations(get_engine())
        logger.info("Database schema ready.")
    except Exception as exc:  # pragma: no cover - startup resilience
        logger.warning("Migration on startup failed (continuing): %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="QuantumRiskLab API",
        version="0.1.0",
        description=(
            "Quantum-enhanced portfolio optimization and risk analysis. "
            "Compares classical (SciPy) and quantum(-inspired)/QUBO approaches."
        ),
        lifespan=lifespan,
    )

    # Permissive CORS for the local dashboard.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(optimize.router)
    app.include_router(portfolio.router)
    app.include_router(experiments.router)
    app.include_router(assets.router)
    app.include_router(analytics.router)

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "name": "QuantumRiskLab",
            "version": "0.1.0",
            "qiskit_available": qiskit_available(),
            "docs": "/docs",
        }

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    # Serve the static dashboard at /dashboard when the directory is present.
    dashboard_dir = Path(__file__).resolve().parents[3] / "dashboard"
    if dashboard_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_dir), html=True),
            name="dashboard",
        )

    return app


app = create_app()
