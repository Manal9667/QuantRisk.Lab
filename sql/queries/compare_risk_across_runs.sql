-- compare_risk_across_runs.sql
-- Compare portfolio risk across optimization runs by joining the solver record
-- (optimization_runs) with its financial risk snapshot (risk_metrics).
-- Optionally scoped to a single experiment.
--
-- Named parameter: :experiment_id (nullable)
SELECT
    r.run_id,
    r.experiment_id,
    r.run_type,
    r.solver,
    r.universe_size,
    r.n_selected,
    r.objective_value,
    r.runtime_seconds,
    rm.expected_return,
    rm.volatility,
    rm.sharpe_ratio,
    rm.max_drawdown,
    rm.beta,
    rm.var_95,
    rm.cvar_95,
    rm.turnover,
    rm.diversification_ratio,
    rm.effective_n,
    rm.max_sector_exposure,
    rm.constraint_violation,
    r.created_at
FROM optimization_runs r
LEFT JOIN risk_metrics rm ON rm.run_id = r.run_id
WHERE (:experiment_id IS NULL OR r.experiment_id = :experiment_id)
ORDER BY r.universe_size, r.run_type, r.created_at;
