-- build_features.sql
-- Builds the features table from raw_prices and raw_macro.
-- Idempotent: CREATE OR REPLACE rebuilds the table from scratch on every run.
-- rsi_14 is intentionally NULL here — populated downstream by build_features.py.
-- All window frames use PRECEDING only. No lookahead bias.

CREATE OR REPLACE TABLE features AS
WITH returns AS (
    SELECT
        ticker,
        date,
        adj_close,
        volume,
        (adj_close - LAG(adj_close) OVER w) / LAG(adj_close) OVER w AS return_1d
    FROM raw_prices
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
windowed AS (
    SELECT
        *,
        CASE WHEN COUNT(*) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) < 5
             THEN NULL
             ELSE AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
        END AS ma_5,
        CASE WHEN COUNT(*) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) < 20
             THEN NULL
             ELSE AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        END AS ma_20,
        CASE WHEN COUNT(*) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) < 50
             THEN NULL
             ELSE AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
        END AS ma_50,
        CASE WHEN COUNT(return_1d) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) < 20
             THEN NULL
             ELSE STDDEV(return_1d) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        END AS volatility_20,
        CASE WHEN COUNT(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) < 20
             THEN NULL
             ELSE (volume - AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))
                  / NULLIF(STDDEV(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)
        END AS volume_zscore,
        CASE
            WHEN LEAD(adj_close) OVER (PARTITION BY ticker ORDER BY date) IS NULL THEN NULL
            WHEN LEAD(adj_close) OVER (PARTITION BY ticker ORDER BY date) > adj_close THEN 1
            ELSE 0
        END::INTEGER AS target
    FROM returns
)
SELECT
    w.ticker,
    w.date,
    w.return_1d,
    w.ma_5,
    w.ma_20,
    w.ma_50,
    w.volatility_20,
    NULL::DOUBLE AS rsi_14,
    w.volume_zscore,
    ff.fedfunds,
    cpi.cpiaucsl,
    vix.vix,
    spread.t10y2y,
    w.target,
    'v1'              AS feature_version,
    current_timestamp AS computed_at
FROM windowed w
ASOF JOIN (SELECT date, value AS fedfunds  FROM raw_macro WHERE series_id = 'FEDFUNDS') ff     ON ff.date    <= w.date - INTERVAL '1' DAY
ASOF JOIN (SELECT date, value AS cpiaucsl  FROM raw_macro WHERE series_id = 'CPIAUCSL') cpi    ON cpi.date   <= w.date - INTERVAL '1' DAY
ASOF JOIN (SELECT date, value AS vix       FROM raw_macro WHERE series_id = 'VIXCLS')   vix    ON vix.date   <= w.date - INTERVAL '1' DAY
ASOF JOIN (SELECT date, value AS t10y2y    FROM raw_macro WHERE series_id = 'T10Y2Y')   spread ON spread.date <= w.date - INTERVAL '1' DAY;
