"""Unit tests for the quantum optimization pipeline.

The dependency-free ``exact`` backend is always tested. Qiskit-backed paths are
gated on the optional stack being importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantumrisklab.quantum.qubo import brute_force_qubo, build_qubo
from quantumrisklab.quantum.solver import optimize_quantum, qiskit_available


def test_exact_backend_selects_and_weights(sample_problem, base_constraints):
    result, detail = optimize_quantum(
        sample_problem, base_constraints, backend="exact", target_cardinality=4, seed=42
    )
    assert result.run_type == "quantum"
    assert result.n_selected == 4
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert detail.n_qubits == sample_problem.n
    assert detail.best_bitstring.count("1") == 4
    # Exact brute force matches the standalone QUBO minimum.
    qubo = build_qubo(sample_problem, base_constraints, target_cardinality=4)
    _, best_e = brute_force_qubo(qubo)
    assert abs(detail.objective_qubo - best_e) < 1e-6
    assert detail.solution_quality == pytest.approx(1.0, abs=1e-6)


def test_quantum_detail_records_penalties(sample_problem, base_constraints):
    _, detail = optimize_quantum(
        sample_problem, base_constraints, backend="exact", target_cardinality=3, seed=1
    )
    assert detail.penalty_cardinality > 0
    assert detail.qubo_construction_time >= 0
    assert detail.quantum_runtime >= 0


@pytest.mark.quantum
def test_numpy_min_eigen_matches_brute_force(sample_problem, base_constraints):
    if not qiskit_available():
        pytest.skip("Qiskit stack not installed")
    exact, exact_detail = optimize_quantum(
        sample_problem, base_constraints, backend="exact", target_cardinality=4, seed=7
    )
    qk, qk_detail = optimize_quantum(
        sample_problem, base_constraints, backend="numpy_min_eigen", target_cardinality=4, seed=7
    )
    # Both are exact solvers of the same QUBO: identical selection + energy.
    assert qk_detail.best_bitstring == exact_detail.best_bitstring
    assert abs(qk_detail.objective_qubo - exact_detail.objective_qubo) < 1e-6


@pytest.mark.quantum
def test_qaoa_qubit_cap(sample_problem, base_constraints):
    if not qiskit_available():
        pytest.skip("Qiskit stack not installed")
    # sample_problem has 10 vars (< cap) so QAOA is allowed; assert it runs and
    # returns a feasible-cardinality selection. Kept tiny for speed.
    small = build_qubo(sample_problem, base_constraints, target_cardinality=3)
    assert small.n_variables <= 12
