# QuantumRiskLab

**A quantum-enhanced portfolio optimization and risk-analysis platform for financial use cases.**

QuantumRiskLab takes historical market data for a universe of 30–100 liquid
equities/ETFs and runs a full workflow: data ingestion and cleaning, financial
feature calculation, portfolio risk measurement, classical portfolio
optimization, a QUBO-based quantum(-inspired) optimizer, and a **side-by-side,
honest comparison** of the two approaches — all backed by a PostgreSQL database
that stores every run so experiments are reproducible and queryable.

This project does **not** claim quantum advantage. Its purpose is to investigate,
experimentally, *where* a quantum optimization formulation could augment
classical financial workflows and *where* classical optimization remains
preferable. For the tested problem sizes, the classical optimizer wins on
solution quality and runtime — see [Results](#results) and
[Discussion](#discussion-does-quantum-beat-classical-here).

## Highlights

- Full-stack research platform: **~3,400 lines of Python** across **49 modules**,
  **327 lines of analytical SQL**, and **43 automated tests** (unit + integration).
- End-to-end pipeline (ingest → clean → feature engineering → optimize → compare)
  over universes of **30–100 assets**, backed by **10 normalized PostgreSQL tables**
  and **6 analytical SQL queries** using window functions and conditional aggregates.
- **4 interchangeable QUBO solver backends** (exact enumeration, simulated
  annealing, exact eigensolver, gate-model QAOA) benchmarked against a SciPy
  mean-variance optimizer with 8 real-world constraints.
- Reproducible, seeded experiment harness sweeping **N = 10 → 75**, recording QUBO
  construction time, solver runtime, objective value, risk, constraint violations,
  and solution quality for every run.
- Key measured finding: the classical optimizer delivered a **higher Sharpe ratio
  at every problem size** (e.g. **1.63 vs 1.00 at N = 75**) while running
  **~20–40× faster**; the QUBO was solved to optimality (quality = 1.0), and
  gate-model QAOA reproduced the exact optimum at N = 8 but **~10⁵× slower
  (130 s vs ~1 ms)**.
- **11 FastAPI endpoints** plus a dependency-free analytics dashboard; one-command
  **Docker Compose** stack (API + PostgreSQL).

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Why this matters to a financial institution](#why-this-matters-to-a-financial-institution)
3. [System architecture](#system-architecture)
4. [Database schema](#database-schema)
5. [SQL examples](#sql-examples)
6. [Mathematical formulation](#mathematical-formulation)
7. [QUBO formulation](#qubo-formulation)
8. [Classical optimization approach](#classical-optimization-approach)
9. [Quantum approach](#quantum-approach)
10. [Experimental methodology](#experimental-methodology)
11. [Results](#results)
12. [Discussion: does quantum beat classical here?](#discussion-does-quantum-beat-classical-here)
13. [Limitations](#limitations)
14. [Getting started](#getting-started)
15. [API reference](#api-reference)
16. [Project layout](#project-layout)
17. [Testing](#testing)

---

## Problem statement

Given a universe of *N* liquid assets and their historical returns, choose a
portfolio that maximises risk-adjusted return subject to realistic constraints
(a cap on the number of holdings, position-size bounds, sector-exposure limits,
a volatility ceiling, a transaction-cost penalty, and an optional beta target).

Two families of methods are implemented and compared:

* **Classical** — continuous mean-variance optimization via SciPy (`SLSQP`), with
  a two-stage heuristic for the cardinality (max-holdings) constraint.
* **Quantum(-inspired)** — the *cardinality-constrained selection* sub-problem is
  cast as a **QUBO** (Quadratic Unconstrained Binary Optimization) and solved
  with (a) exact enumeration, (b) simulated annealing, (c) Qiskit's
  `NumPyMinimumEigensolver`, and (d) gate-model **QAOA** on a statevector
  simulator.

The central research question:

> **When does a quantum optimization formulation provide a potentially useful
> way to model financial portfolio problems, and where does classical
> optimization remain preferable?**

---

## Why this matters to a financial institution

Portfolio construction and risk measurement are core to asset management, wealth
management, treasury, and trading desks at a bank such as RBC:

* **Capital efficiency** — allocating a limited risk budget across assets to
  maximise risk-adjusted return is a daily activity for portfolio managers.
* **Risk and regulatory reporting** — volatility, Value-at-Risk (VaR),
  Conditional VaR (CVaR), beta, drawdown, and sector concentration feed risk
  limits and regulatory capital.
* **Cost-aware rebalancing** — turnover and transaction costs materially erode
  returns; optimizers must trade off expected improvement against trading cost.
* **Combinatorial structure** — real mandates impose *discrete* rules (hold at
  most *K* names, minimum lot sizes, index-tracking baskets). These are exactly
  the constraints where a binary/QUBO formulation is natural — and where the
  question of quantum relevance is most interesting.

Quantum optimization is an active research area at large banks. A credible
contribution is **not** a hype claim, but a rigorous, reproducible harness that
measures *when* the quantum formulation helps and when it does not.

---

## System architecture

```
                +------------------------------------------------------+
                |                    FastAPI backend                   |
                |  /optimize/classical  /optimize/quantum  /assets ... |
                +-------------------------+----------------------------+
                                          |
        +-----------------+   +-----------v-----------+   +----------------+
        |  Data pipeline  |   |   Application service |   |   Dashboard    |
        | ingest/clean/   |   |  build problem ->     |   | (static, Chart |
        | synth generate  |   |  optimize -> persist  |   |  .js, /dashboard)|
        +--------+--------+   +-----+-----------+-----+   +----------------+
                 |                  |           |
        +--------v--------+  +------v----+  +---v-----------------+
        |    Features     |  | Classical |  | Quantum / QUBO      |
        | returns, cov,   |  | optimizer |  | build_qubo + solve  |
        | beta, risk      |  | (SciPy)   |  | (exact/SA/QAOA/NPME) |
        +--------+--------+  +-----+-----+  +----------+----------+
                 |                 |                   |
                 +--------+--------+---------+---------+
                          |                  |
                 +--------v------------------v--------+
                 |            PostgreSQL              |
                 | assets, market_prices, daily_returns,
                 | portfolio_constraints, experiments,
                 | optimization_runs, portfolio_weights,
                 | risk_metrics, quantum_runs         |
                 +-----------------------------------+
```

The database is a **first-class part of the architecture**, not a passive store:
analytical SQL (window functions, conditional aggregates, cross-run joins)
computes rolling volatility, most-frequently-selected assets, and the
classical-vs-quantum comparison directly in the database.

**Tech stack:** Python 3.11+ · NumPy/SciPy/Pandas · SQLAlchemy 2 + PostgreSQL
(psycopg 3) · Qiskit + qiskit-optimization + qiskit-algorithms · FastAPI ·
Docker Compose · Chart.js dashboard.

> **Note on the local environment.** This repository was developed and verified
> end-to-end on **CPython 3.14** using SQLite as a zero-infrastructure fallback,
> with the full Qiskit stack installed via `abi3` wheels. The Docker image pins
> **Python 3.11 + PostgreSQL 16** for the production-style deployment. The same
> SQLAlchemy models drive both backends.

---

## Database schema

Eight normalised tables (plus an `experiments` grouping table and a
`schema_migrations` bookkeeping table). Full DDL is in
[`migrations/001_initial_schema.sql`](migrations/001_initial_schema.sql).

| Table | Purpose | Key columns |
|---|---|---|
| `assets` | Investable universe | `asset_id` PK, `ticker` (unique), `sector`, `asset_class` |
| `market_prices` | Daily OHLCV | `asset_id` FK, `price_date`, `adj_close`; unique `(asset_id, price_date)` |
| `daily_returns` | Simple + log returns | `asset_id` FK, `return_date`, `simple_return`, `log_return` |
| `portfolio_constraints` | Reusable constraint/objective set | `max_holdings`, `min/max_weight`, `max_volatility`, `max_sector_exposure`, `transaction_cost`, `risk_aversion`, `target_beta` |
| `experiments` | A sweep grouping many runs | `experiment_id` PK, `seed`, `name` |
| `optimization_runs` | One solver execution | `run_type` (`classical`/`quantum`), `solver`, `universe_size`, `n_variables`, `objective_value`, `runtime_seconds` |
| `portfolio_weights` | Resulting allocation | `run_id` FK, `asset_id` FK, `weight`, `selected` |
| `risk_metrics` | Financial risk snapshot (1:1 with run) | `expected_return`, `volatility`, `sharpe_ratio`, `max_drawdown`, `beta`, `var_95`, `cvar_95`, `turnover`, `diversification_ratio`, `effective_n`, `constraint_violation` |
| `quantum_runs` | QUBO/circuit detail (extends a quantum run) | `backend`, `n_qubits`, `qubo_construction_time`, `quantum_runtime`, `penalty_cardinality`, `objective_qubo`, `solution_quality`, `best_bitstring`, `num_feasible_samples` |

**Design choice:** `optimization_runs` records *solver execution* metadata
(objective value, runtime, solver), while `risk_metrics` holds the *financial
risk* of the resulting portfolio. Separating them keeps the objective/runtime
comparison clean and lets risk be compared across runs via a single join.

---

## SQL examples

All analytical queries live in [`sql/queries/`](sql/queries) and are executed by
[`QueryLibrary`](src/quantumrisklab/db/queries.py). Examples:

**Rolling annualised volatility** (portable windowed sample-variance identity):

```sql
SELECT a.ticker, r.return_date,
  SQRT(CASE WHEN v < 0.0 THEN 0.0 ELSE v END) * SQRT(252) AS rolling_vol_annualized
FROM ( SELECT r.*,
         ( SUM(simple_return*simple_return) OVER w
           - {window} * POWER(AVG(simple_return) OVER w, 2) ) / ({window}-1.0) AS v
       FROM daily_returns r
       WINDOW w AS (PARTITION BY asset_id ORDER BY return_date
                    ROWS BETWEEN {window_minus_1} PRECEDING AND CURRENT ROW) ) r ...
```

**Classical vs quantum, per (experiment, size)** — conditional aggregates put the
two run types on one row:

```sql
SELECT r.universe_size,
  MAX(r.objective_value) FILTER (WHERE r.run_type='classical') AS classical_objective,
  MAX(r.objective_value) FILTER (WHERE r.run_type='quantum')   AS quantum_objective,
  AVG(r.runtime_seconds) FILTER (WHERE r.run_type='classical') AS classical_runtime_s,
  AVG(r.runtime_seconds) FILTER (WHERE r.run_type='quantum')   AS quantum_runtime_s
FROM optimization_runs r
GROUP BY r.universe_size ORDER BY r.universe_size;
```

**Most frequently selected assets** across all runs:

```sql
SELECT a.ticker, a.sector,
  COUNT(*) FILTER (WHERE pw.selected) AS times_selected,
  (1.0 * COUNT(*) FILTER (WHERE pw.selected)) / NULLIF(COUNT(DISTINCT r.run_id),0) AS selection_frequency
FROM portfolio_weights pw
JOIN assets a ON a.asset_id = pw.asset_id
JOIN optimization_runs r ON r.run_id = pw.run_id
GROUP BY a.asset_id ORDER BY times_selected DESC;
```

Also provided: `historical_returns` (returns computed in SQL via `LAG`),
`compare_risk_across_runs`, and `optimization_performance_over_time`.

---

## Mathematical formulation

Let `w ∈ ℝ^N` be portfolio weights, `μ` annualised expected returns, `Σ`
annualised covariance, `β` per-asset betas, and `w₀` previous weights.

**Objective (utility, maximised):**

```
U(w) = μᵀw  −  λ · (wᵀ Σ w)  −  c · ‖w − w₀‖₁
       └ return ┘  └── risk ──┘   └ transaction cost ┘
```

* `λ` = risk aversion, `c` = proportional transaction cost.

**Constraints:**

```
Σ wᵢ = 1                          (fully invested / budget)
0 ≤ wᵢ ≤ w_max        and  wᵢ ∈ {0} ∪ [w_min, w_max]   (box + min lot if held)
|{ i : wᵢ > 0 }| ≤ K              (cardinality / max holdings)
√(wᵀ Σ w) ≤ σ_max                 (volatility cap)
Σ_{i∈s} wᵢ ≤ e_max   ∀ sector s   (sector exposure cap)
|βᵀw − β*| ≤ δ                    (optional beta band)
```

**Reported metrics:** expected return, volatility, Sharpe ratio
`(μᵀw − r_f)/σ`, max drawdown, beta, 1-day historical VaR/CVaR at 95%, turnover,
transaction cost, diversification ratio, effective number of holdings
`1/Σwᵢ²`, and sector exposures. See
[`features/risk.py`](src/quantumrisklab/features/risk.py).

`U(w)` is reported as the run's `objective_value` for **both** classical and
quantum solutions, so they are compared on a common footing.

---

## QUBO formulation

The **cardinality-constrained selection** sub-problem uses binary variables
`xᵢ ∈ {0,1}` ("asset *i* is selected"). Energy to **minimise**:

```
E(x) = − θ Σ μᵢ xᵢ                       (reward expected return)
       + q Σ_{i,j} Σᵢⱼ xᵢ xⱼ             (penalise portfolio risk)
       + P_card ( Σ xᵢ − K )²            (enforce exactly K holdings)
       + P_div  Σ_{i<j, sec(i)=sec(j)} xᵢ xⱼ   (discourage sector concentration)
```

Rewritten as `E(x) = xᵀ Q x + offset` using the binary identity `xᵢ² = xᵢ`:

* return: `Q_ii += −θ μᵢ`
* risk: `Q += q Σ`
* cardinality: `Q_ii += P_card(1 − 2K)`, `Q_ij += P_card` (i≠j), `offset += P_card K²`
* diversification: `Q_ij += P_div/2` for same-sector pairs

Penalties `P_card`, `P_div` are auto-scaled from the data so the cardinality
constraint dominates the objective terms; whether a solution actually satisfies
the constraint is **measured** (via `constraint_violation`), never assumed. See
[`quantum/qubo.py`](src/quantumrisklab/quantum/qubo.py).

Because the QUBO decides only *selection*, the selected names are then weighted
with the **same** continuous solver used classically — isolating the selection
decision so the comparison is about selection quality, not weighting.

---

## Classical optimization approach

Cardinality-constrained mean-variance optimization is a mixed-integer quadratic
program (MIQP). SciPy has no integer solver, so cardinality is handled with a
transparent **two-stage heuristic**
([`optimization/classical.py`](src/quantumrisklab/optimization/classical.py)):

1. **Select** — solve the continuous problem over the full universe with all
   convex constraints (budget, box, volatility, sector, beta); rank by weight.
2. **Weight** — if more than `K` names are held, keep the top-`K` and re-solve
   weights over that subset, enforcing the minimum position size.

This is a strong, fast baseline that mirrors practitioner workflows. It is a
heuristic, not a global MIQP solve — stated honestly.

---

## Quantum approach

Four interchangeable backends solve the **same** QUBO
([`quantum/solver.py`](src/quantumrisklab/quantum/solver.py)):

| Backend | What it is | Scales to |
|---|---|---|
| `exact` | Dependency-free brute-force enumeration (ground truth) | ≤ 18 qubits |
| `exact` (auto) | Simulated annealing with cardinality-preserving swap moves | any N (quantum-inspired) |
| `numpy_min_eigen` | Qiskit `MinimumEigenOptimizer` + `NumPyMinimumEigensolver` (exact Ising diagonalisation) | ≲ 20 qubits |
| `qaoa` | Gate-model **QAOA** on Qiskit's `StatevectorSampler` primitive | ≤ 12 qubits (simulator) |

Simulated annealing is the classical analogue of quantum annealing (the paradigm
behind D-Wave-style hardware) and is the backend used for the scaling sweep.
QAOA demonstrates the gate-model route but is capped at small sizes because a
statevector simulator scales as `2ⁿ`.

---

## Experimental methodology

Everything is **seeded and reproducible**. The sweep
([`experiments/runner.py`](src/quantumrisklab/experiments/runner.py)):

1. Generates a deterministic synthetic universe from a linear factor model
   (market + sector factors + idiosyncratic noise), giving realistic covariance
   structure and cross-sectional beta dispersion. No live feeds; no hardcoded
   results — only *generative parameters* are set.
2. For each size in `{10, 20, 30, 50, 75}` builds the problem on a stable subset,
   sets `K = round(0.30·N)`, runs the classical optimizer and the QUBO solver,
   and (at small sizes) QAOA.
3. Records **every requested quantity** into PostgreSQL: problem size, number of
   variables, QUBO construction time, classical optimization time, quantum
   runtime, objective value, portfolio risk, constraint violations, and solution
   quality.
4. Exports a tidy CSV/JSON to `results/` for the dashboard and this README.

Reproduce with:

```bash
python scripts/run_experiments.py --sizes 10 20 30 50 75
```

**Solution quality** for a quantum run is `reference_QUBO_energy / obtained_QUBO_energy`
(1.0 = matched the best-known energy). It measures how well the solver minimised
the QUBO — separately from whether that QUBO is a good model of the portfolio
objective.

---

## Results

Synthetic universe, **seed = 42**, 756 trading days, `K = round(0.30·N)`,
`λ = 5`, transaction cost = 10 bps, `w_max = 0.35`, sector cap = 0.50. Quantum
backend = `exact` (brute force at N=10, simulated annealing above). Numbers are
from a single reproducible run of the experiment script.

### Classical vs quantum(-inspired), by size

| N | K | Cl. return | Q. return | Cl. vol | Q. vol | Cl. Sharpe | Q. Sharpe | Cl. U(w) | Q. U(w) | Cl. time | Q. time | Q. solver |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 10 | 3 | 0.101 | 0.026 | 0.200 | 0.173 | 0.403 | 0.035 | −0.100 | −0.124 | 0.08 s | 0.01 s | brute force |
| 20 | 6 | 0.385 | 0.171 | 0.231 | 0.183 | 1.577 | 0.824 | +0.116 | +0.002 | 0.15 s | 7.7 s | ann. |
| 30 | 9 | 0.384 | 0.208 | 0.224 | 0.186 | 1.623 | 1.008 | +0.131 | +0.033 | 0.27 s | 12.8 s | ann. |
| 50 | 15 | 0.378 | 0.207 | 0.220 | 0.178 | 1.622 | 1.050 | +0.134 | +0.047 | 0.32 s | 11.3 s | ann. |
| 75 | 22 | 0.361 | 0.186 | 0.210 | 0.166 | 1.627 | 0.996 | +0.140 | +0.047 | 0.54 s | 13.5 s | ann. |

* **Constraint violations were 0.0 for every run** (both approaches feasible).
* **Solution quality = 1.0 for every quantum run** — the QUBO was solved to
  optimality at all sizes.

### Gate-model QAOA (n = 8, K = 3)

| Solver | Best bitstring | QUBO energy | Solution quality | Runtime | Feasible samples |
|:--|:--:|--:|--:|--:|--:|
| `exact` (brute force) | `00001101` | 1.6005 | 1.000 | 0.001 s | 1 / 1 |
| `qaoa` (statevector) | `00001101` | 1.6005 | 1.000 | **130 s** | 54 / 1024 |

QAOA found the **identical optimum** as brute force — at roughly **10⁵× the
runtime** — and cannot scale past ~12 qubits on a laptop simulator.

---

## Discussion: does quantum beat classical here?

**No — not for the tested problem sizes, and this project does not claim it does.**
The evidence is nuanced and worth stating precisely:

1. **The classical optimizer dominates on the financial objective.** At every
   size it delivers higher expected return, higher Sharpe, and higher utility
   `U(w)` than the quantum(-inspired) selection.

2. **The gap is a *formulation* gap, not a *solver* gap.** The QUBO was solved to
   optimality (quality = 1.0) at every size, and QAOA matched brute force
   exactly at n = 8. So the quantum solver is not "getting the wrong answer" — it
   is answering a **coarser question**. Binary selection plus quadratic penalties
   is a lower-fidelity model of the true continuous, constrained mean-variance
   problem than the SLSQP formulation. Even a *perfect* QUBO solve inherits that
   modelling loss.

3. **The quantum-inspired portfolios are systematically lower-volatility / more
   diversified.** That is a direct artifact of the diversification penalty and
   equal-ish binary selection, not evidence of a risk advantage.

4. **Runtime strongly favours classical.** Continuous optimization runs in tens
   to hundreds of milliseconds; simulated annealing on the QUBO takes 8–14 s at
   these sizes, and gate-model QAOA is ~10⁵× slower still on a simulator.

**So where *could* a quantum/QUBO formulation be useful?**

* **Genuinely discrete/combinatorial objectives** that convex solvers model
  poorly — hard cardinality, minimum lot sizes, index-tracking baskets,
  buy-in thresholds — where the QUBO is the *natural* model rather than a lossy
  approximation.
* **Very large discrete instances on purpose-built annealing hardware**, where a
  physical annealer could sample low-energy states faster than exhaustive
  classical search (untested here; requires real hardware).
* **As a rigorous baseline harness** — this project's real contribution is a
  reproducible way to *measure* the gap rather than assert an outcome.

For continuous, well-behaved mean-variance portfolio construction at 10–75
assets, **classical optimization remains clearly preferable.**

---

## Limitations

* **Synthetic data.** Returns come from a seeded factor model, not live market
  data. The ingestion layer is abstracted ([`data/ingestion.py`](src/quantumrisklab/data/ingestion.py))
  so a real provider can be dropped in, but no market feed is bundled.
* **Classical cardinality is heuristic**, not a global MIQP solve.
* **QAOA runs on a statevector simulator** with the reference sampler (no
  `qiskit-aer`, no real QPU); it is capped at 12 qubits and is slow.
* **QUBO ≠ full mean-variance problem.** The QUBO models selection with fixed
  penalties; it does not reproduce the continuous objective exactly.
* **Single seed reported.** The script is fully reproducible; multi-seed
  averaging is left as an extension.
* **No transaction-lot / integer-share modelling** beyond the cardinality
  constraint.

---

## Getting started

### Option A — Docker (API + PostgreSQL)

```bash
cp .env.example .env         # optional; sensible defaults are baked in
docker compose up --build
```

* API:        http://localhost:8000
* Dashboard:  http://localhost:8000/dashboard
* API docs:   http://localhost:8000/docs

The API container waits for PostgreSQL, runs migrations, seeds a synthetic
universe (idempotent), and serves the app.

Run the experiment sweep inside the stack:

```bash
docker compose run --rm api experiments --sizes 10 20 30 50 75
```

### Option B — Local Python (zero infrastructure, SQLite)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-quantum.txt
pip install -e .

# Use SQLite so no database server is needed:
$env:QRL_DATABASE_URL = "sqlite:///./quantumrisklab.db"   # PowerShell
# export QRL_DATABASE_URL=sqlite:///./quantumrisklab.db   # bash

python scripts/init_db.py --seed-data
python scripts/run_experiments.py --sizes 10 20 30 50 75 --no-qaoa
uvicorn quantumrisklab.api.main:app --reload
```

The quantum stack is optional: without it, the `exact` backend (brute force /
simulated annealing) still runs; only `qaoa` and `numpy_min_eigen` require Qiskit.

### Configuration

All settings are environment variables prefixed `QRL_` (see
[`.env.example`](.env.example) and [`config.py`](src/quantumrisklab/config.py)):
`QRL_DATABASE_URL`, `QRL_RANDOM_SEED`, `QRL_N_ASSETS`, `QRL_N_TRADING_DAYS`,
`QRL_RISK_AVERSION`, `QRL_TRANSACTION_COST`, `QRL_QUANTUM_BACKEND`, etc.

---

## API reference

| Method | Path | Description |
|---|---|---|
| POST | `/optimize/classical` | Run the classical optimizer; persist and return the portfolio + metrics |
| POST | `/optimize/quantum` | Run the QUBO optimizer (`backend`: `exact`/`qaoa`/`numpy_min_eigen`) |
| GET | `/portfolio/{run_id}` | Retrieve a persisted portfolio (weights) |
| GET | `/risk/{run_id}` | Retrieve the risk snapshot for a run |
| GET | `/experiments` | List experiments |
| GET | `/experiments/compare` | Classical-vs-quantum, performance-over-time, risk-across-runs (SQL) |
| GET | `/assets` | List the investable universe |
| GET | `/analytics/rolling-volatility` | Rolling annualised volatility (SQL) |
| GET | `/analytics/most-selected` | Most frequently selected assets (SQL) |
| GET | `/analytics/correlation` | Correlation matrix for the dashboard heatmap |

Example:

```bash
curl -X POST http://localhost:8000/optimize/quantum \
  -H "Content-Type: application/json" \
  -d '{"universe_size":20,"backend":"exact","target_cardinality":6,
       "constraints":{"max_holdings":6,"min_weight":0.02,"max_weight":0.35,
                      "risk_aversion":5.0,"max_sector_exposure":0.5}}'
```

---

## Project layout

```
migrations/            SQL migrations (PostgreSQL DDL)
sql/queries/           Analytical SQL (rolling vol, comparisons, selection freq.)
src/quantumrisklab/
  config.py            Env-driven settings
  domain.py            Core dataclasses (problem, constraints, metrics, results, QUBO)
  db/                  SQLAlchemy models, connection, migrations, repository, query loader
  data/                Ingestion, cleaning, deterministic synthetic generation, pipeline
  features/            Returns, risk metrics, problem construction
  optimization/        Classical optimizer, shared objective + weight solver
  quantum/             QUBO builder + solvers (exact/SA/QAOA/NumPyMinEigen)
  experiments/         Reproducible sweep runner
  service.py           Application service (load -> optimize -> persist)
  api/                 FastAPI app, schemas, routes
dashboard/             Static Chart.js dashboard (served at /dashboard)
scripts/               init_db.py, run_experiments.py
tests/                 unit/ + integration/
docker/                entrypoint.sh
Dockerfile, docker-compose.yml
```

---

## Testing

```bash
pip install -r requirements-dev.txt
$env:PYTHONPATH = "src"          # or: pip install -e .
pytest -q                        # unit + integration (SQLite, no services needed)
pytest -q -m "not quantum"       # skip Qiskit-dependent tests
```

The suite (43 tests) covers: deterministic data generation, return/risk math,
QUBO construction and solvers, the classical optimizer, the DB pipeline and
analytical SQL, the experiment sweep, and every API endpoint — all on an
ephemeral SQLite database.

---

## License

MIT.
