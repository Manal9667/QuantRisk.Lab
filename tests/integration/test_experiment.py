"""Integration test for the reproducible experiment sweep."""

from __future__ import annotations

import pytest

from quantumrisklab.db.connection import session_scope
from quantumrisklab.db.queries import QueryLibrary
from quantumrisklab.experiments.runner import ExperimentConfig, run_experiment_sweep

pytestmark = pytest.mark.integration


def test_sweep_persists_and_compares(tmp_db):
    config = ExperimentConfig(
        name="test-sweep",
        sizes=[8, 10],
        seed=42,
        n_trading_days=260,
        run_qaoa=False,
    )
    summary = run_experiment_sweep(config)

    assert summary["experiment_id"] is not None
    assert len(summary["rows"]) == 2
    for row in summary["rows"]:
        # Every requested metric was recorded.
        assert row["n_variables"] == row["size"]
        assert row["qubo_construction_time_s"] >= 0
        assert row["classical_time_s"] >= 0
        assert row["quantum_time_s"] >= 0
        assert row["classical_objective"] is not None
        assert row["quantum_objective"] is not None
        assert row["quantum_solution_quality"] is not None

    # The SQL comparison should now return one row per size (2 sizes).
    ql = QueryLibrary(tmp_db)
    comparison = ql.classical_vs_quantum(summary["experiment_id"])
    assert len(comparison) == 2
    for c in comparison:
        assert c["classical_objective"] is not None
        assert c["quantum_objective"] is not None

    # Runs, weights, and risk metrics were persisted (2 sizes x 2 run types).
    perf = ql.optimization_performance_over_time(summary["experiment_id"])
    assert len(perf) == 4  # {classical, quantum} x {8, 10}


def test_most_selected_after_sweep(tmp_db):
    config = ExperimentConfig(name="sel", sizes=[10], seed=42, n_trading_days=260, run_qaoa=False)
    run_experiment_sweep(config)
    ql = QueryLibrary(tmp_db)
    rows = ql.most_selected_assets(limit=10)
    assert len(rows) > 0
    assert rows[0]["times_selected"] >= 1
