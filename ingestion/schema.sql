-- Daily OHLCV price data ingested from external sources (e.g. Yahoo Finance)
CREATE TABLE IF NOT EXISTS raw_prices (
    ticker      VARCHAR   NOT NULL,
    date        DATE      NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE    NOT NULL,
    volume      BIGINT,
    ingested_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, date)
);

-- Macroeconomic time-series data ingested from external sources (e.g. FRED)
CREATE TABLE IF NOT EXISTS raw_macro (
    series_id   VARCHAR   NOT NULL,
    date        DATE      NOT NULL,
    value       DOUBLE,
    ingested_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (series_id, date)
);

-- Audit log tracking each ingestion run, its status, and any errors encountered
CREATE TABLE IF NOT EXISTS ingestion_log (
    run_id      VARCHAR   NOT NULL,
    source      VARCHAR   NOT NULL,
    ticker      VARCHAR,
    rows_loaded INTEGER,
    status      VARCHAR,
    error_msg   VARCHAR,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP
);
