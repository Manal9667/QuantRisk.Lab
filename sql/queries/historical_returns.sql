-- historical_returns.sql
-- Calculate historical simple and log returns directly from stored prices using
-- a window function, rather than trusting a precomputed table. Optionally scoped
-- to a single ticker and/or date range. Demonstrates return calculation in SQL.
--
-- Named parameters: :ticker (nullable), :start_date (nullable), :end_date (nullable)
SELECT
    a.ticker,
    a.sector,
    mp.price_date,
    mp.adj_close,
    LAG(mp.adj_close) OVER w AS prev_adj_close,
    mp.adj_close / NULLIF(LAG(mp.adj_close) OVER w, 0) - 1.0        AS simple_return,
    LN(mp.adj_close / NULLIF(LAG(mp.adj_close) OVER w, 0))          AS log_return
FROM market_prices mp
JOIN assets a ON a.asset_id = mp.asset_id
WHERE (:ticker IS NULL OR a.ticker = :ticker)
  AND (:start_date IS NULL OR mp.price_date >= :start_date)
  AND (:end_date   IS NULL OR mp.price_date <= :end_date)
WINDOW w AS (PARTITION BY a.asset_id ORDER BY mp.price_date)
ORDER BY a.ticker, mp.price_date;
