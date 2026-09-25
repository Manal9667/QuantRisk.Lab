"""Unit tests for the classical optimizer and shared weight solver."""

from __future__ import annotations

import numpy as np

from quantumrisklab.domain import PortfolioConstraints
from quantumrisklab.optimization.classical import optimize_classical
from quantumrisklab.optimization.objective import portfolio_utility, solve_weights


def test_weights_sum_to_one(sample_problem, base_constraints):
    result = optimize_classical(sample_problem, base_constraints)
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_respects_cardinality(sample_problem, base_constraints):
    result = optimize_classical(sample_problem, base_constraints)
    assert result.n_selected <= base_constraints.max_holdings


def test_respects_weight_bounds(sample_problem, base_constraints):
    result = optimize_classical(sample_problem, base_constraints)
    held = result.weights[result.selected]
    assert (held <= base_constraints.max_weight + 1e-6).all()
    assert (held >= base_constraints.min_weight - 1e-6).all()


def test_metrics_present_and_feasible(sample_problem, base_constraints):
    result = optimize_classical(sample_problem, base_constraints)
    assert result.metrics is not None
    # A converged classical solution should be (near) feasible.
    assert result.metrics.constraint_violation < 1e-3


def test_higher_risk_aversion_reduces_volatility(sample_problem):
    low = PortfolioConstraints(name="low", max_holdings=6, min_weight=0.0, max_weight=1.0, risk_aversion=0.5)
    high = PortfolioConstraints(name="high", max_holdings=6, min_weight=0.0, max_weight=1.0, risk_aversion=50.0)
    r_low = optimize_classical(sample_problem, low)
    r_high = optimize_classical(sample_problem, high)
    assert r_high.metrics.volatility <= r_low.metrics.volatility + 1e-6


def test_solve_weights_single_asset(sample_problem, base_constraints):
    w = solve_weights(sample_problem, base_constraints, subset=[2])
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[2] - 1.0) < 1e-9


def test_portfolio_utility_definition(sample_problem):
    c = PortfolioConstraints(name="u", risk_aversion=3.0, transaction_cost=0.0)
    n = sample_problem.n
    w = np.full(n, 1.0 / n)
    expected = float(w @ sample_problem.expected_returns) - 3.0 * float(w @ sample_problem.cov_matrix @ w)
    assert abs(portfolio_utility(w, sample_problem, c) - expected) < 1e-9
