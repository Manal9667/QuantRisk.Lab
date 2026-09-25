"""Unit tests for deterministic synthetic data generation."""

from __future__ import annotations

import numpy as np
import pytest

from quantumrisklab.data.synthetic import generate_universe


def test_generation_is_deterministic():
    u1 = generate_universe(8, 200, seed=123)
    u2 = generate_universe(8, 200, seed=123)
    np.testing.assert_allclose(u1.prices.to_numpy(), u2.prices.to_numpy())
    assert u1.tickers == u2.tickers


def test_different_seeds_differ():
    u1 = generate_universe(8, 200, seed=1)
    u2 = generate_universe(8, 200, seed=2)
    assert not np.allclose(u1.prices.to_numpy(), u2.prices.to_numpy())


def test_shapes_and_positive_prices():
    u = generate_universe(15, 300, seed=42)
    assert u.prices.shape == (300, 15)
    assert len(u.tickers) == len(u.sectors) == len(u.names) == 15
    assert (u.prices.to_numpy() > 0).all()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_universe(0, 100, seed=1)
    with pytest.raises(ValueError):
        generate_universe(10, 10, seed=1)
