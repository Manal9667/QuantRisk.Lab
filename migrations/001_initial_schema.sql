-- ===========================================================================
-- QuantumRiskLab :: initial schema (migration 001)
-- PostgreSQL dialect. Applied by src/quantumrisklab/db/migrate.py.
--
-- Design notes
-- ------------
-- The database is a first-class part of the architecture, not a dump store:
--   * assets / market_prices / daily_returns hold the cleaned data pipeline
--     output and feed feature calculation.
--   * portfolio_constraints captures a reusable, named constraint set.
--   * experiments group a sweep of optimization runs (e.g. 10..75 assets).
--   * optimization_runs records ONE solver execution (classical or quantum)
--     with its objective value and runtime; it is the spine of the comparisons.
--   * portfolio_weights holds the resulting allocation (one row per asset).
--   * risk_metrics holds the financial risk snapshot of a run (1:1 with a run),
--     kept separate from solver metadata so risk can be compared across runs.
--   * quantum_runs extends a quantum optimization_run with QUBO/circuit detail.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- assets: the investable universe (equities / ETFs).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id      BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(16)  NOT NULL UNIQUE,
    name          VARCHAR(128) NOT NULL,
    sector        VARCHAR(64)  NOT NULL,
    asset_class   VARCHAR(32)  NOT NULL DEFAULT 'equity',
    currency      VARCHAR(8)   NOT NULL DEFAULT 'USD',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_assets_sector ON assets (sector);

-- ---------------------------------------------------------------------------
-- market_prices: daily OHLCV. adj_close drives return calculation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_prices (
    price_id      BIGSERIAL PRIMARY KEY,
    asset_id      BIGINT       NOT NULL REFERENCES assets (asset_id) ON DELETE CASCADE,
    price_date    DATE         NOT NULL,
    open          DOUBLE PRECISION,
    high          DOUBLE PRECISION,
    low           DOUBLE PRECISION,
    close         DOUBLE PRECISION NOT NULL,
    adj_close     DOUBLE PRECISION NOT NULL,
    volume        BIGINT,
    CONSTRAINT uq_market_prices UNIQUE (asset_id, price_date)
);
CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices (price_date);
CREATE INDEX IF NOT EXISTS idx_market_prices_asset_date ON market_prices (asset_id, price_date);

-- ---------------------------------------------------------------------------
-- daily_returns: simple and log returns derived from adj_close.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_returns (
    return_id      BIGSERIAL PRIMARY KEY,
    asset_id       BIGINT      NOT NULL REFERENCES assets (asset_id) ON DELETE CASCADE,
    return_date    DATE        NOT NULL,
    simple_return  DOUBLE PRECISION NOT NULL,
    log_return     DOUBLE PRECISION NOT NULL,
    CONSTRAINT uq_daily_returns UNIQUE (asset_id, return_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_returns_date ON daily_returns (return_date);
CREATE INDEX IF NOT EXISTS idx_daily_returns_asset_date ON daily_returns (asset_id, return_date);

-- ---------------------------------------------------------------------------
-- portfolio_constraints: a named, reusable constraint / objective set.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_constraints (
    constraint_id        BIGSERIAL PRIMARY KEY,
    name                 VARCHAR(128) NOT NULL,
    max_holdings         INTEGER,               -- cardinality cap (K)
    min_weight           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    max_weight           DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    max_volatility       DOUBLE PRECISION,      -- annualised vol cap
    max_sector_exposure  DOUBLE PRECISION,      -- per-sector weight cap
    transaction_cost     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    risk_aversion        DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    target_beta          DOUBLE PRECISION,      -- optional beta target
    beta_tolerance       DOUBLE PRECISION,      -- allowed +/- band around target
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- experiments: a batch/sweep grouping many optimization runs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  BIGSERIAL PRIMARY KEY,
    name           VARCHAR(128) NOT NULL,
    seed           INTEGER      NOT NULL,
    description    TEXT,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- optimization_runs: one solver execution (classical or quantum).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id             BIGSERIAL PRIMARY KEY,
    experiment_id      BIGINT REFERENCES experiments (experiment_id) ON DELETE SET NULL,
    constraint_id      BIGINT REFERENCES portfolio_constraints (constraint_id) ON DELETE SET NULL,
    run_type           VARCHAR(16) NOT NULL CHECK (run_type IN ('classical', 'quantum')),
    solver             VARCHAR(64) NOT NULL,
    universe_size      INTEGER     NOT NULL,   -- N: number of candidate assets
    n_variables        INTEGER     NOT NULL,   -- decision variables in the model
    n_selected         INTEGER,                -- holdings in the resulting portfolio
    objective_value    DOUBLE PRECISION,       -- solver objective (comparable per problem)
    runtime_seconds    DOUBLE PRECISION NOT NULL,
    status             VARCHAR(24) NOT NULL DEFAULT 'completed',
    seed               INTEGER,
    notes              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON optimization_runs (experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_type ON optimization_runs (run_type);
CREATE INDEX IF NOT EXISTS idx_runs_universe ON optimization_runs (universe_size);

-- ---------------------------------------------------------------------------
-- portfolio_weights: allocation produced by a run (one row per asset).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_weights (
    weight_id   BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES optimization_runs (run_id) ON DELETE CASCADE,
    asset_id    BIGINT NOT NULL REFERENCES assets (asset_id) ON DELETE CASCADE,
    weight      DOUBLE PRECISION NOT NULL,
    selected    BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_portfolio_weights UNIQUE (run_id, asset_id)
);
CREATE INDEX IF NOT EXISTS idx_weights_run ON portfolio_weights (run_id);
CREATE INDEX IF NOT EXISTS idx_weights_asset ON portfolio_weights (asset_id);
CREATE INDEX IF NOT EXISTS idx_weights_selected ON portfolio_weights (selected);

-- ---------------------------------------------------------------------------
-- risk_metrics: financial risk snapshot of a run's portfolio (1:1 with run).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_metrics (
    metric_id             BIGSERIAL PRIMARY KEY,
    run_id                BIGINT NOT NULL UNIQUE REFERENCES optimization_runs (run_id) ON DELETE CASCADE,
    expected_return       DOUBLE PRECISION,     -- annualised
    volatility            DOUBLE PRECISION,     -- annualised
    sharpe_ratio          DOUBLE PRECISION,
    max_drawdown          DOUBLE PRECISION,
    beta                  DOUBLE PRECISION,
    var_95                DOUBLE PRECISION,     -- 1-day historical VaR (positive = loss)
    cvar_95               DOUBLE PRECISION,     -- 1-day historical CVaR
    turnover              DOUBLE PRECISION,
    transaction_cost_paid DOUBLE PRECISION,
    diversification_ratio DOUBLE PRECISION,
    effective_n           DOUBLE PRECISION,     -- inverse Herfindahl of weights
    max_sector_exposure   DOUBLE PRECISION,
    constraint_violation  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- quantum_runs: QUBO / circuit detail extending a quantum optimization_run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS quantum_runs (
    quantum_run_id          BIGSERIAL PRIMARY KEY,
    run_id                  BIGINT NOT NULL UNIQUE REFERENCES optimization_runs (run_id) ON DELETE CASCADE,
    backend                 VARCHAR(32) NOT NULL,   -- exact | qaoa | numpy_min_eigen
    n_qubits                INTEGER NOT NULL,
    n_variables             INTEGER NOT NULL,
    qubo_construction_time  DOUBLE PRECISION NOT NULL,
    quantum_runtime         DOUBLE PRECISION NOT NULL,
    penalty_cardinality     DOUBLE PRECISION,
    penalty_diversification DOUBLE PRECISION,
    reps                    INTEGER,                -- QAOA depth p
    shots                   INTEGER,
    best_bitstring          VARCHAR(256),
    objective_qubo          DOUBLE PRECISION,       -- QUBO objective of best sample
    num_feasible_samples    INTEGER,
    solution_quality        DOUBLE PRECISION,       -- ratio vs best-known / classical
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
