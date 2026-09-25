"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quantumrisklab.api.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def client(seeded_db):
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_root_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    root = client.get("/").json()
    assert root["name"] == "QuantumRiskLab"


def test_assets_endpoint(client):
    assets = client.get("/assets").json()
    assert len(assets) == 10
    assert {"ticker", "sector", "name"} <= set(assets[0])


def test_classical_optimization_endpoint(client):
    payload = {
        "universe_size": 10,
        "constraints": {"max_holdings": 4, "min_weight": 0.05, "max_weight": 0.4, "risk_aversion": 5.0},
    }
    r = client.post("/optimize/classical", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["run_type"] == "classical"
    assert body["n_selected"] <= 4
    assert body["metrics"] is not None
    assert body["run_id"] is not None


def test_quantum_optimization_and_retrieval(client):
    payload = {
        "universe_size": 10,
        "backend": "exact",
        "target_cardinality": 4,
        "constraints": {"max_holdings": 4, "min_weight": 0.05, "max_weight": 0.4, "risk_aversion": 5.0},
    }
    r = client.post("/optimize/quantum", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["run_type"] == "quantum"
    assert body["quantum_detail"]["backend"] == "exact"
    run_id = body["run_id"]

    # Retrieve the persisted portfolio and risk.
    portfolio = client.get(f"/portfolio/{run_id}")
    assert portfolio.status_code == 200
    assert len(portfolio.json()["weights"]) >= 1

    risk = client.get(f"/risk/{run_id}")
    assert risk.status_code == 200
    assert risk.json()["volatility"] is not None


def test_missing_portfolio_returns_404(client):
    assert client.get("/portfolio/999999").status_code == 404


def test_experiments_and_compare(client):
    # Run one of each type so comparison has content.
    payload = {"universe_size": 10, "constraints": {"max_holdings": 4}}
    client.post("/optimize/classical", json=payload)
    client.post("/optimize/quantum", json={**payload, "backend": "exact", "target_cardinality": 4})

    assert client.get("/experiments").status_code == 200
    compare = client.get("/experiments/compare")
    assert compare.status_code == 200
    assert "classical_vs_quantum" in compare.json()


def test_analytics_endpoints(client):
    assert client.get("/analytics/rolling-volatility?window=21&ticker=AST000").status_code == 200
    assert client.get("/analytics/most-selected").status_code == 200
    corr = client.get("/analytics/correlation?universe_size=8").json()
    assert len(corr["tickers"]) == 8
    assert len(corr["matrix"]) == 8
