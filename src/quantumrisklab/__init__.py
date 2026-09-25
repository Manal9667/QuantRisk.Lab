"""QuantumRiskLab.

A quantum-enhanced portfolio optimization and risk analysis platform.

The package is organised into cohesive layers:

* ``config``        -- environment-driven settings.
* ``db``            -- SQLAlchemy models, session management, repository, SQL loader.
* ``data``          -- ingestion, cleaning, and deterministic synthetic generation.
* ``features``      -- return and risk feature calculation.
* ``optimization``  -- classical portfolio optimizer, constraints, metrics.
* ``quantum``       -- QUBO formulation and Qiskit-backed solvers.
* ``experiments``   -- reproducible experiment orchestration.
* ``api``           -- FastAPI backend.

The design goal is to compare, experimentally and honestly, classical and
quantum(-inspired) approaches to cardinality-constrained portfolio selection.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
