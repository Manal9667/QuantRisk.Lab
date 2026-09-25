-- most_selected_assets.sql
-- Find the assets selected most frequently across all optimization runs
-- (optionally filtered by run_type). Surfaces which names the optimizers
-- persistently favour. A run "selects" an asset when portfolio_weights.selected
-- is TRUE.
--
-- Named parameters: :run_type (nullable), :limit
SELECT
    a.ticker,
    a.name,
    a.sector,
    COUNT(*) FILTER (WHERE pw.selected)                        AS times_selected,
    COUNT(DISTINCT r.run_id)                                   AS runs_considered,
    (1.0 * COUNT(*) FILTER (WHERE pw.selected))
        / NULLIF(COUNT(DISTINCT r.run_id), 0)                  AS selection_frequency,
    AVG(pw.weight) FILTER (WHERE pw.selected)                  AS avg_weight_when_selected
FROM portfolio_weights pw
JOIN assets a            ON a.asset_id = pw.asset_id
JOIN optimization_runs r ON r.run_id = pw.run_id
WHERE (:run_type IS NULL OR r.run_type = :run_type)
GROUP BY a.asset_id, a.ticker, a.name, a.sector
HAVING COUNT(*) FILTER (WHERE pw.selected) > 0
ORDER BY times_selected DESC, selection_frequency DESC
LIMIT :limit;
