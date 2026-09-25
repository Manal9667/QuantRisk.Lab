"""Quantum solver dispatch and the quantum optimization pipeline.

Three interchangeable backends solve the SAME QUBO:

  * ``exact``           -- dependency-free classical enumeration (ground truth for
                           small n) with a simulated-annealing fallback for large n.
  * ``numpy_min_eigen`` -- Qiskit ``MinimumEigenOptimizer`` + ``NumPyMinimumEigensolver``
                           (exact diagonalisation of the Ising Hamiltonian).
  * ``qaoa``            -- Qiskit QAOA on the reference ``StatevectorSampler`` primitive.

``optimize_quantum`` builds the QUBO, solves the *selection*, then weights the
selected names with the SAME continuous solver used classically, so classical and
quantum are compared on selection quality with weighting held constant.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from quantumrisklab.domain import (
    OptimizationResult,
    PortfolioConstraints,
    PortfolioProblem,
    QuantumSolveDetail,
    QuboArtifacts,
)
from quantumrisklab.features.risk import portfolio_metrics
from quantumrisklab.optimization.objective import portfolio_utility, solve_weights
from quantumrisklab.quantum.qubo import (
    brute_force_qubo,
    build_qubo,
    simulated_annealing_qubo,
)

BRUTE_FORCE_LIMIT = 18
# The reference StatevectorSampler makes gate-model QAOA impractical beyond a
# handful of qubits (n=8 ~ 100s on a laptop). We cap it so a QAOA run cannot
# accidentally hang an experiment; this limitation is a documented finding.
QAOA_QUBIT_LIMIT = 12


def qiskit_available() -> bool:
    """Return True if the optional Qiskit optimization stack is importable."""
    try:
        import qiskit_algorithms  # noqa: F401
        import qiskit_optimization  # noqa: F401
    except Exception:
        return False
    return True


def _qubo_to_quadratic_program(qubo: QuboArtifacts):
    """Build a Qiskit ``QuadraticProgram`` from our symmetric QUBO matrix."""
    from qiskit_optimization import QuadraticProgram

    n = qubo.n_variables
    Q = qubo.Q
    qp = QuadraticProgram("portfolio_selection_qubo")
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    # x^T Q x = sum_i Q_ii x_i + 2 * sum_{i<j} Q_ij x_i x_j  (binary).
    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}
    quadratic: dict[tuple[str, str], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            coeff = float(Q[i, j] + Q[j, i])  # = 2*Q_ij for symmetric Q
            if coeff != 0.0:
                quadratic[(f"x{i}", f"x{j}")] = coeff
    qp.minimize(constant=float(qubo.offset), linear=linear, quadratic=quadratic)
    return qp


def _solve_qiskit(
    qubo: QuboArtifacts,
    backend: str,
    reps: int,
    shots: int,
    seed: int,
    qaoa_maxiter: int = 50,
) -> tuple[np.ndarray, float, int]:
    """Solve via a Qiskit MinimumEigenOptimizer. Returns (x, energy, n_feasible)."""
    from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    algorithm_globals.random_seed = seed
    qp = _qubo_to_quadratic_program(qubo)

    if backend == "qaoa":
        if qubo.n_variables > QAOA_QUBIT_LIMIT:
            raise RuntimeError(
                f"QAOA on the reference simulator is capped at {QAOA_QUBIT_LIMIT} "
                f"qubits (requested {qubo.n_variables}). Use backend 'exact' "
                "(brute force / simulated annealing) at this size."
            )
        sampler = StatevectorSampler(seed=seed)
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=qaoa_maxiter), reps=reps)
        meo = MinimumEigenOptimizer(qaoa)
    elif backend == "numpy_min_eigen":
        meo = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown qiskit backend: {backend}")

    result = meo.solve(qp)
    x = np.asarray(result.x, dtype=float)
    energy = float(result.fval)

    # Count how many returned samples satisfy the target cardinality.
    n_feasible = 0
    samples = getattr(result, "samples", None) or []
    for s in samples:
        if int(round(np.asarray(s.x).sum())) == qubo.target_cardinality:
            n_feasible += 1
    if not samples and int(round(x.sum())) == qubo.target_cardinality:
        n_feasible = 1

    return x, energy, n_feasible


def _reference_energy(qubo: QuboArtifacts, seed: int) -> float:
    """Best-known QUBO energy for scoring solution quality."""
    if qubo.n_variables <= BRUTE_FORCE_LIMIT:
        _, e = brute_force_qubo(qubo)
        return e
    _, e = simulated_annealing_qubo(qubo, seed=seed)
    return e


def optimize_quantum(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    backend: str = "exact",
    target_cardinality: Optional[int] = None,
    reps: int = 1,
    shots: int = 1024,
    seed: int = 42,
    risk_free_rate: float = 0.02,
    penalty_cardinality: Optional[float] = None,
    penalty_diversification: Optional[float] = None,
    qaoa_maxiter: int = 50,
    compute_metrics: bool = True,
    compute_quality: bool = True,
) -> tuple[OptimizationResult, QuantumSolveDetail]:
    """Run the quantum selection pipeline and return (result, quantum_detail)."""
    qubo = build_qubo(
        problem,
        constraints,
        target_cardinality=target_cardinality,
        penalty_cardinality=penalty_cardinality,
        penalty_diversification=penalty_diversification,
    )

    n = problem.n
    solve_start = time.perf_counter()

    if backend in ("qaoa", "numpy_min_eigen"):
        if not qiskit_available():
            raise RuntimeError(
                f"backend '{backend}' requires the Qiskit stack "
                "(pip install -r requirements-quantum.txt)."
            )
        x, qubo_energy, n_feasible = _solve_qiskit(
            qubo, backend, reps, shots, seed, qaoa_maxiter=qaoa_maxiter
        )
        solver_name = f"qiskit_{backend}"
        used_reps = reps if backend == "qaoa" else None
        used_shots = shots if backend == "qaoa" else None
    elif backend == "exact":
        if n <= BRUTE_FORCE_LIMIT:
            x, qubo_energy = brute_force_qubo(qubo)
            solver_name = "qubo_bruteforce_exact"
        else:
            x, qubo_energy = simulated_annealing_qubo(qubo, seed=seed)
            solver_name = "qubo_simulated_annealing"
        n_feasible = 1 if int(round(x.sum())) == qubo.target_cardinality else 0
        used_reps = None
        used_shots = None
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    quantum_runtime = time.perf_counter() - solve_start

    selected_idx = np.where(x > 0.5)[0]
    # Weight the selected names with the shared continuous solver.
    if len(selected_idx) > 0:
        weights = solve_weights(
            problem, constraints, subset=selected_idx, enforce_min_weight=True, seed=seed
        )
    else:
        weights = np.zeros(n)
    selected = weights > 1e-6

    objective_value = portfolio_utility(weights, problem, constraints)
    metrics = (
        portfolio_metrics(weights, problem, constraints, risk_free_rate)
        if compute_metrics
        else None
    )

    # Solution quality vs best-known QUBO energy (energies are positive here due
    # to the large cardinality-penalty offset, so the ratio lies in (0, 1]).
    solution_quality: Optional[float] = None
    if compute_quality:
        ref_energy = _reference_energy(qubo, seed)
        if qubo_energy > 0 and ref_energy > 0:
            solution_quality = float(ref_energy / qubo_energy)
        elif abs(qubo_energy - ref_energy) < 1e-6:
            solution_quality = 1.0

    best_bitstring = "".join(str(int(round(b))) for b in x)

    result = OptimizationResult(
        run_type="quantum",
        solver=solver_name,
        tickers=list(problem.tickers),
        weights=weights,
        selected=selected,
        objective_value=objective_value,
        runtime_seconds=quantum_runtime + qubo.construction_time_seconds,
        universe_size=n,
        n_variables=qubo.n_variables,
        metrics=metrics,
        notes=f"backend={backend}; target_K={qubo.target_cardinality}; selected={len(selected_idx)}",
    )

    detail = QuantumSolveDetail(
        backend=backend,
        n_qubits=qubo.n_variables,
        n_variables=qubo.n_variables,
        qubo_construction_time=qubo.construction_time_seconds,
        quantum_runtime=quantum_runtime,
        penalty_cardinality=qubo.penalty_cardinality,
        penalty_diversification=qubo.penalty_diversification,
        reps=used_reps,
        shots=used_shots,
        best_bitstring=best_bitstring,
        objective_qubo=qubo_energy,
        num_feasible_samples=n_feasible,
        solution_quality=solution_quality,
    )

    return result, detail
