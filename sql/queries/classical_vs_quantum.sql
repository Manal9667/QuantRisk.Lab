-- classical_vs_quantum.sql
-- Compare classical vs quantum solutions side by side, per (experiment,
-- universe_size). Aggregates the two run_types onto a single row so objective,
-- risk, and runtime can be diffed directly. Uses conditional aggregates rather
-- than a self-join so it tolerates missing counterparts gracefully.
--
-- Named parameter: :experiment_id (nullable)
SELECT
    r.experiment_id,
    r.universe_size,
    MAX(r.objective_value) FILTER (WHERE r.run_type = 'classical') AS classical_objective,
    MAX(r.objective_value) FILTER (WHERE r.run_type = 'quantum')   AS quantum_objective,
    MAX(rm.volatility)     FILTER (WHERE r.run_type = 'classical') AS classical_volatility,
    MAX(rm.volatility)     FILTER (WHERE r.run_type = 'quantum')   AS quantum_volatility,
    MAX(rm.expected_return) FILTER (WHERE r.run_type = 'classical') AS classical_return,
    MAX(rm.expected_return) FILTER (WHERE r.run_type = 'quantum')   AS quantum_return,
    MAX(rm.sharpe_ratio)   FILTER (WHERE r.run_type = 'classical') AS classical_sharpe,
    MAX(rm.sharpe_ratio)   FILTER (WHERE r.run_type = 'quantum')   AS quantum_sharpe,
    AVG(r.runtime_seconds) FILTER (WHERE r.run_type = 'classical') AS classical_runtime_s,
    AVG(r.runtime_seconds) FILTER (WHERE r.run_type = 'quantum')   AS quantum_runtime_s,
    MAX(rm.constraint_violation) FILTER (WHERE r.run_type = 'quantum') AS quantum_violation
FROM optimization_runs r
LEFT JOIN risk_metrics rm ON rm.run_id = r.run_id
WHERE (:experiment_id IS NULL OR r.experiment_id = :experiment_id)
GROUP BY r.experiment_id, r.universe_size
ORDER BY r.experiment_id, r.universe_size;
