-- rolling_volatility.sql
-- Retrieve rolling annualised volatility per asset from daily_returns.
-- The rolling standard deviation is computed with a windowed sample-variance
-- identity (portable across PostgreSQL and SQLite):
--     var = (Sum(x^2) - n * mean^2) / (n - 1)
-- then annualised by sqrt(trading_days_per_year).
--
-- {window} is an integer window length (validated in Python before formatting;
-- it is NOT free user text, so this is not an injection surface).
-- {ann} is the annualisation factor (trading days per year).
-- Named parameter: :ticker (nullable)
SELECT
    a.ticker,
    r.return_date,
    r.simple_return,
    SQRT(
        CASE
            WHEN (
                SUM(r.simple_return * r.simple_return) OVER w
                - {window} * POWER(AVG(r.simple_return) OVER w, 2)
            ) / ({window} - 1.0) < 0.0
            THEN 0.0
            ELSE (
                SUM(r.simple_return * r.simple_return) OVER w
                - {window} * POWER(AVG(r.simple_return) OVER w, 2)
            ) / ({window} - 1.0)
        END
    ) * SQRT({ann}) AS rolling_vol_annualized
FROM daily_returns r
JOIN assets a ON a.asset_id = r.asset_id
WHERE (:ticker IS NULL OR a.ticker = :ticker)
WINDOW w AS (
    PARTITION BY r.asset_id
    ORDER BY r.return_date
    ROWS BETWEEN {window_minus_1} PRECEDING AND CURRENT ROW
)
ORDER BY a.ticker, r.return_date;
