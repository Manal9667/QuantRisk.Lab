"""Unit tests for the QUBO formulation and dependency-free solvers."""

from __future__ import annotations

import numpy as np

from quantumrisklab.domain import QuboArtifacts
from quantumrisklab.quantum.qubo import (
    brute_force_qubo,
    build_qubo,
    simulated_annealing_qubo,
)


def test_qubo_is_symmetric(sample_problem, base_constraints):
    qubo = build_qubo(sample_problem, base_constraints, target_cardinality=4)
    np.testing.assert_allclose(qubo.Q, qubo.Q.T, atol=1e-12)
    assert qubo.n_variables == sample_problem.n
    assert qubo.target_cardinality == 4
    assert qubo.penalty_cardinality > 0


def test_energy_matches_manual():
    Q = np.array([[1.0, 0.5], [0.5, 2.0]])
    qubo = QuboArtifacts(
        Q=Q, offset=3.0, n_variables=2, penalty_cardinality=0.0,
        penalty_diversification=0.0, construction_time_seconds=0.0, target_cardinality=1,
    )
    x = np.array([1.0, 1.0])
    # x^T Q x + offset = (1+0.5+0.5+2) + 3 = 7
    assert abs(qubo.energy(x) - 7.0) < 1e-9


def test_brute_force_finds_minimum():
    Q = np.array([[-1.0, 0.0], [0.0, -2.0]])
    qubo = QuboArtifacts(
        Q=Q, offset=0.0, n_variables=2, penalty_cardinality=0.0,
        penalty_diversification=0.0, construction_time_seconds=0.0, target_cardinality=2,
    )
    x, e = brute_force_qubo(qubo)
    np.testing.assert_allclose(x, [1.0, 1.0])
    assert abs(e - (-3.0)) < 1e-9


def test_cardinality_penalty_enforces_k(sample_problem, base_constraints):
    # With a strong cardinality penalty, the brute-force optimum selects exactly K.
    qubo = build_qubo(sample_problem, base_constraints, target_cardinality=3)
    x, _ = brute_force_qubo(qubo)
    assert int(x.sum()) == 3


def test_simulated_annealing_matches_brute_force_small(sample_problem, base_constraints):
    qubo = build_qubo(sample_problem, base_constraints, target_cardinality=3)
    bx, be = brute_force_qubo(qubo)
    sx, se = simulated_annealing_qubo(qubo, seed=42, n_restarts=15, n_steps=1500)
    # SA should reach the global optimum on a 10-variable instance.
    assert abs(se - be) < 1e-6
    assert int(sx.sum()) == 3
