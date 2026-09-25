"""Reproducible experiment sweep.

For each problem size in a configured list, the runner:
  1. builds the optimization instance from the seeded universe (a stable subset),
  2. runs the classical optimizer,
  3. runs the quantum(-inspired) QUBO solver (``exact`` -> brute force for small n,
     simulated annealing for large n),
  4. optionally runs gate-model QAOA at sizes within the simulator's reach,
  5. records every requested metric (problem size, #variables, QUBO build time,
     classical time, quantum runtime, objective value, risk, constraint
     violations, solution quality) into the database, and
  6. exports a tidy results table to ``results/`` for the dashboard and README.

Everything is seeded, so re-running reproduces the results bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from quantumrisklab.config import get_settings
from quantumrisklab.data.pipeline import seed_market_data
from quantumrisklab.db.connection import get_engine, session_scope
from quantumrisklab.db.migrate import run_migrations
from quantumrisklab.db.queries import QueryLibrary
from quantumrisklab.db.repository import (
    create_constraint,
    create_experiment,
    load_returns_frame,
    persist_result,
    sector_map,
)
from quantumrisklab.domain import PortfolioConstraints
from quantumrisklab.features.problem import build_problem_from_frame
from quantumrisklab.optimization.classical import optimize_classical
from quantumrisklab.quantum.solver import QAOA_QUBIT_LIMIT, optimize_quantum, qiskit_available

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment sweep."""

    name: str = "size-sweep"
    sizes: list[int] = field(default_factory=lambda: [10, 20, 30, 50, 75])
    seed: int = 42
    n_trading_days: int = 756
    risk_aversion: float = 5.0
    transaction_cost: float = 0.001
    min_weight: float = 0.02
    # Must satisfy K_min * max_weight >= 1 for the smallest case to be feasible
    # (with cardinality_fraction=0.30 the smallest K is 3, so 3 * 0.35 = 1.05).
    max_weight: float = 0.35
    max_sector_exposure: float = 0.50
    max_volatility: Optional[float] = None
    cardinality_fraction: float = 0.30  # K = round(fraction * size), min 3
    run_qaoa: bool = True
    qaoa_max_size: int = QAOA_QUBIT_LIMIT
    qaoa_maxiter: int = 50
    risk_free_rate: float = 0.02

    def cardinality_for(self, size: int) -> int:
        return max(3, min(size, round(self.cardinality_fraction * size)))


def _ensure_database(max_size: int, seed: int, n_trading_days: int) -> None:
    """Migrate and seed a universe large enough for the largest experiment."""
    settings = get_settings()
    # Override generation params to guarantee enough assets/history.
    settings_seed = settings.model_copy(
        update={"n_assets": max_size, "n_trading_days": n_trading_days, "random_seed": seed}
    )
    engine = get_engine()
    run_migrations(engine)
    seed_market_data(settings=settings_seed, force=False)


def run_experiment_sweep(config: ExperimentConfig) -> dict:
    """Execute the sweep and return a summary dict (also written to results/)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    max_size = max(config.sizes)
    _ensure_database(max_size, config.seed, config.n_trading_days)

    with session_scope() as session:
        returns = load_returns_frame(session)
        sectors = sector_map(session)
        all_tickers = list(returns.columns)
        if len(all_tickers) < max_size:
            raise RuntimeError(
                f"Universe has {len(all_tickers)} assets but sweep needs {max_size}."
            )

        experiment = create_experiment(
            session,
            name=config.name,
            seed=config.seed,
            description=f"Classical vs quantum sweep over sizes {config.sizes}",
        )
        experiment_id = experiment.experiment_id

        rows: list[dict] = []
        for size in config.sizes:
            subset = all_tickers[:size]
            problem = build_problem_from_frame(returns, sectors, tickers=subset)
            K = config.cardinality_for(size)
            constraints = PortfolioConstraints(
                name=f"{config.name}-n{size}-K{K}",
                max_holdings=K,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
                max_sector_exposure=config.max_sector_exposure,
                max_volatility=config.max_volatility,
                transaction_cost=config.transaction_cost,
                risk_aversion=config.risk_aversion,
            )
            constraint_row = create_constraint(session, constraints)
            constraint_id = constraint_row.constraint_id

            logger.info("size=%d K=%d: classical...", size, K)
            classical = optimize_classical(
                problem, constraints, risk_free_rate=config.risk_free_rate, seed=config.seed
            )
            persist_result(
                session,
                classical,
                experiment_id=experiment_id,
                constraint_id=constraint_id,
                seed=config.seed,
            )

            logger.info("size=%d K=%d: quantum-inspired (exact/annealing)...", size, K)
            q_result, q_detail = optimize_quantum(
                problem,
                constraints,
                backend="exact",
                target_cardinality=K,
                seed=config.seed,
                risk_free_rate=config.risk_free_rate,
            )
            persist_result(
                session,
                q_result,
                experiment_id=experiment_id,
                constraint_id=constraint_id,
                seed=config.seed,
                quantum_detail=q_detail,
            )

            row = {
                "experiment_id": experiment_id,
                "size": size,
                "cardinality_K": K,
                "n_variables": problem.n,
                "qubo_construction_time_s": q_detail.qubo_construction_time,
                "classical_time_s": classical.runtime_seconds,
                "quantum_time_s": q_detail.quantum_runtime,
                "classical_objective": classical.objective_value,
                "quantum_objective": q_result.objective_value,
                "classical_volatility": classical.metrics.volatility,
                "quantum_volatility": q_result.metrics.volatility,
                "classical_return": classical.metrics.expected_return,
                "quantum_return": q_result.metrics.expected_return,
                "classical_sharpe": classical.metrics.sharpe_ratio,
                "quantum_sharpe": q_result.metrics.sharpe_ratio,
                "classical_violation": classical.metrics.constraint_violation,
                "quantum_violation": q_result.metrics.constraint_violation,
                "quantum_solution_quality": q_detail.solution_quality,
                "quantum_backend": q_detail.backend,
                "quantum_solver": q_result.solver,
            }

            # Optional gate-model QAOA at small sizes only.
            if config.run_qaoa and problem.n <= config.qaoa_max_size and qiskit_available():
                logger.info("size=%d: QAOA (gate-model, simulator)...", size)
                qa_result, qa_detail = optimize_quantum(
                    problem,
                    constraints,
                    backend="qaoa",
                    target_cardinality=K,
                    seed=config.seed,
                    qaoa_maxiter=config.qaoa_maxiter,
                    risk_free_rate=config.risk_free_rate,
                )
                persist_result(
                    session,
                    qa_result,
                    experiment_id=experiment_id,
                    constraint_id=constraint_id,
                    seed=config.seed,
                    quantum_detail=qa_detail,
                )
                row.update(
                    {
                        "qaoa_time_s": qa_detail.quantum_runtime,
                        "qaoa_objective": qa_result.objective_value,
                        "qaoa_solution_quality": qa_detail.solution_quality,
                        "qaoa_violation": qa_result.metrics.constraint_violation,
                    }
                )

            rows.append(row)

    # Persisted; now export tidy results (outside the write transaction).
    df = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RESULTS_DIR / f"experiment_{experiment_id}_{stamp}.csv"
    json_path = RESULTS_DIR / f"experiment_{experiment_id}_{stamp}.json"
    df.to_csv(csv_path, index=False)

    engine = get_engine()
    ql = QueryLibrary(engine)
    comparison = ql.classical_vs_quantum(experiment_id)
    performance = ql.optimization_performance_over_time(experiment_id)

    summary = {
        "experiment_id": experiment_id,
        "config": {
            "name": config.name,
            "sizes": config.sizes,
            "seed": config.seed,
            "cardinality_fraction": config.cardinality_fraction,
        },
        "rows": rows,
        "sql_classical_vs_quantum": comparison,
        "sql_performance_over_time": performance,
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote results to %s and %s", csv_path, json_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the classical-vs-quantum experiment sweep.")
    parser.add_argument("--name", default="size-sweep")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50, 75],
        help="Problem sizes (number of candidate assets) to sweep.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Master seed (defaults to config).")
    parser.add_argument("--no-qaoa", action="store_true", help="Skip the gate-model QAOA runs.")
    parser.add_argument(
        "--qaoa-max-size",
        type=int,
        default=QAOA_QUBIT_LIMIT,
        help="Maximum size at which to attempt QAOA (simulator limit).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    config = ExperimentConfig(
        name=args.name,
        sizes=args.sizes,
        seed=args.seed if args.seed is not None else settings.random_seed,
        n_trading_days=settings.n_trading_days,
        risk_aversion=settings.risk_aversion,
        transaction_cost=settings.transaction_cost,
        run_qaoa=not args.no_qaoa,
        qaoa_max_size=args.qaoa_max_size,
        risk_free_rate=settings.risk_free_rate,
    )
    summary = run_experiment_sweep(config)

    print("\n=== Experiment summary ===")
    print(f"experiment_id = {summary['experiment_id']}")
    for row in summary["rows"]:
        print(
            f"  n={row['size']:>3} K={row['cardinality_K']:>2} | "
            f"classical U={row['classical_objective']:+.4f} "
            f"({row['classical_time_s']*1000:.0f} ms) | "
            f"quantum U={row['quantum_objective']:+.4f} "
            f"({row['quantum_time_s']*1000:.0f} ms, {row['quantum_solver']}) | "
            f"q_quality={row.get('quantum_solution_quality')}"
        )
    print(f"\nResults CSV: {summary['csv_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
