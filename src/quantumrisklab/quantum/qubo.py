"""QUBO formulation of cardinality-constrained portfolio selection.

Decision variables: binary ``x_i in {0, 1}`` = "asset i is selected".

Energy to MINIMISE (x is the binary selection vector):

    E(x) = - theta * sum_i mu_i x_i                    # reward expected return
           + q     * sum_{i,j} Sigma_ij x_i x_j        # penalise portfolio risk
           + P_card * (sum_i x_i - K)^2                # enforce cardinality == K
           + P_div  * sum_{i<j, same sector} x_i x_j   # discourage concentration

Written as ``E(x) = x^T Q x + offset`` with a symmetric matrix ``Q`` using the
binary identity ``x_i^2 = x_i``:

    return term:        Q_ii += -theta * mu_i
    risk term:          Q_ij += q * Sigma_ij           (all i, j incl. diagonal)
    cardinality term:   Q_ii += P_card * (1 - 2K);  Q_ij(i<j) += P_card  (sym)
                        offset += P_card * K^2
    diversification:    Q_ij(i<j, same sector) += P_div / 2   (symmetric)

Penalty magnitudes are auto-scaled from the data so the cardinality constraint
dominates the objective terms unless overridden. Whether the resulting QUBO is
actually satisfied is measured (not assumed) via ``constraint_violation``.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional

import numpy as np

from quantumrisklab.domain import PortfolioConstraints, PortfolioProblem, QuboArtifacts


def _auto_penalties(
    mu: np.ndarray, sigma: np.ndarray, risk_aversion: float, theta: float
) -> tuple[float, float]:
    """Derive default cardinality and diversification penalties from data scale."""
    return_scale = theta * float(np.max(np.abs(mu))) if mu.size else 1.0
    risk_scale = risk_aversion * float(np.max(np.sum(np.abs(sigma), axis=1))) if sigma.size else 1.0
    base = max(return_scale + risk_scale, 1e-6)
    penalty_cardinality = 10.0 * base
    penalty_diversification = 0.5 * base
    return penalty_cardinality, penalty_diversification


def build_qubo(
    problem: PortfolioProblem,
    constraints: PortfolioConstraints,
    target_cardinality: Optional[int] = None,
    theta: float = 1.0,
    penalty_cardinality: Optional[float] = None,
    penalty_diversification: Optional[float] = None,
) -> QuboArtifacts:
    """Construct the QUBO for cardinality-constrained selection.

    ``target_cardinality`` defaults to ``constraints.max_holdings`` (or a sensible
    fraction of the universe when unspecified).
    """
    start = time.perf_counter()
    n = problem.n
    mu = problem.expected_returns
    sigma = problem.cov_matrix
    q = constraints.risk_aversion

    if target_cardinality is None:
        target_cardinality = constraints.max_holdings or max(1, n // 2)
    K = int(min(max(target_cardinality, 1), n))

    p_card, p_div = _auto_penalties(mu, sigma, q, theta)
    if penalty_cardinality is not None:
        p_card = float(penalty_cardinality)
    if penalty_diversification is not None:
        p_div = float(penalty_diversification)

    Q = np.zeros((n, n), dtype=float)

    # Return term (diagonal).
    for i in range(n):
        Q[i, i] += -theta * mu[i]

    # Risk term (full matrix).
    Q += q * sigma

    # Cardinality penalty.
    for i in range(n):
        Q[i, i] += p_card * (1.0 - 2.0 * K)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += p_card
            Q[j, i] += p_card
    offset = p_card * (K**2)

    # Diversification penalty (same-sector pairs).
    for i in range(n):
        for j in range(i + 1, n):
            if problem.sectors[i] == problem.sectors[j]:
                Q[i, j] += p_div / 2.0
                Q[j, i] += p_div / 2.0

    construction_time = time.perf_counter() - start

    return QuboArtifacts(
        Q=Q,
        offset=offset,
        n_variables=n,
        penalty_cardinality=p_card,
        penalty_diversification=p_div,
        construction_time_seconds=construction_time,
        target_cardinality=K,
    )


# --------------------------------------------------------------------------- #
# Dependency-free classical QUBO solvers (ground truth + heuristic fallback)
# --------------------------------------------------------------------------- #
def brute_force_qubo(qubo: QuboArtifacts, max_qubits: int = 22) -> tuple[np.ndarray, float]:
    """Exact QUBO minimisation by enumeration. Feasible only for small ``n``.

    Returns ``(best_bitstring, best_energy)``. Raises for ``n`` above
    ``max_qubits`` to avoid an intractable 2^n enumeration.
    """
    n = qubo.n_variables
    if n > max_qubits:
        raise ValueError(
            f"brute_force_qubo refuses n={n} > {max_qubits} (2^n intractable). "
            "Use simulated_annealing_qubo or a Qiskit backend."
        )
    best_x = np.zeros(n)
    best_e = np.inf
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=float)
        e = qubo.energy(x)
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, float(best_e)


def simulated_annealing_qubo(
    qubo: QuboArtifacts,
    seed: int = 42,
    n_restarts: int = 30,
    n_steps: int = 3000,
    t_start: float = 1.0,
    t_end: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Simulated-annealing QUBO minimiser (heuristic, for larger ``n``).

    A dependency-free, deterministic-by-seed classical heuristic used when exact
    enumeration is infeasible. It is *structure-aware*: because the cardinality
    penalty dominates the landscape, each restart is initialised with exactly
    ``target_cardinality`` ones and the proposal distribution mixes
    cardinality-preserving swap moves (turn one selected bit off and one
    unselected bit on) with occasional single-bit flips. This is the classical
    analogue of quantum annealing and provides a strong reference energy for
    scoring quantum solution quality.
    """
    rng = np.random.default_rng(seed)
    n = qubo.n_variables
    Q = qubo.Q
    k = int(min(max(qubo.target_cardinality, 0), n))

    def energy(x: np.ndarray) -> float:
        return float(x @ Q @ x + qubo.offset)

    def random_k_state() -> np.ndarray:
        x = np.zeros(n)
        if 0 < k < n:
            x[rng.choice(n, size=k, replace=False)] = 1.0
        elif k >= n:
            x[:] = 1.0
        return x

    best_x = random_k_state()
    best_e = energy(best_x)

    schedule = np.geomspace(t_start, t_end, n_steps)
    for _ in range(n_restarts):
        x = random_k_state()
        e = energy(x)
        for t in schedule:
            x_new = x.copy()
            ones = np.flatnonzero(x_new > 0.5)
            zeros = np.flatnonzero(x_new <= 0.5)
            # Prefer cardinality-preserving swaps when both sides are non-empty.
            if rng.random() < 0.8 and ones.size and zeros.size:
                x_new[rng.choice(ones)] = 0.0
                x_new[rng.choice(zeros)] = 1.0
            else:
                i = int(rng.integers(0, n))
                x_new[i] = 1.0 - x_new[i]
            e_new = energy(x_new)
            if e_new < e or rng.random() < np.exp(-(e_new - e) / max(t, 1e-9)):
                x, e = x_new, e_new
                if e < best_e:
                    best_e, best_x = e, x.copy()
    return best_x, float(best_e)
