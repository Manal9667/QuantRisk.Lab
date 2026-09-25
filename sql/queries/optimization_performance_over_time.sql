-- optimization_performance_over_time.sql
-- Track optimization performance over time: how runtime and objective evolve as
-- problem size grows and as runs accumulate. Grouped by run_type and
-- universe_size so scalability trends are visible.
--
-- Named parameter: :experiment_id (nullable)
SELECT
    r.run_type,
    r.universe_size,
    COUNT(*)                          AS n_runs,
    AVG(r.runtime_seconds)            AS avg_runtime_s,
    MIN(r.runtime_seconds)            AS min_runtime_s,
    MAX(r.runtime_seconds)            AS max_runtime_s,
    AVG(r.objective_value)            AS avg_objective,
    AVG(r.n_selected)                 AS avg_holdings,
    MIN(r.created_at)                 AS first_run_at,
    MAX(r.created_at)                 AS last_run_at
FROM optimization_runs r
WHERE (:experiment_id IS NULL OR r.experiment_id = :experiment_id)
GROUP BY r.run_type, r.universe_size
ORDER BY r.universe_size, r.run_type;
