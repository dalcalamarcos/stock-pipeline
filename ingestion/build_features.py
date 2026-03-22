import os
import logging
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/opt/airflow/data/stock_pipeline.duckdb")
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,SPY,QQQ").split(",")

SQL_PATH = Path(__file__).parent.parent / "features" / "build_features.sql"


def compute_rsi(series, window=14):
    """Compute RSI from a Pandas Series of return_1d values."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def run_build():
    """Orchestrate feature computation: run SQL transforms then compute and write RSI."""
    conn = duckdb.connect(DUCKDB_PATH)

    with open(SQL_PATH) as f:
        conn.execute(f.read())

    df = conn.execute(
        "SELECT ticker, date, return_1d FROM features ORDER BY ticker, date"
    ).df()

    rsi_parts = []
    for ticker, group in df.groupby("ticker"):
        rsi_values = compute_rsi(group["return_1d"])
        rsi_parts.append(
            pd.DataFrame(
                {"ticker": ticker, "date": group["date"].values, "rsi_14": rsi_values.values}
            )
        )
    rsi_df = pd.concat(rsi_parts, ignore_index=True)

    conn.execute("CREATE TEMP TABLE rsi_update AS SELECT * FROM rsi_df")
    conn.execute("""
        UPDATE features
        SET rsi_14 = rsi_update.rsi_14
        FROM rsi_update
        WHERE features.ticker = rsi_update.ticker
          AND features.date = rsi_update.date
    """)
    conn.execute("DROP TABLE rsi_update")

    summary = conn.execute("""
        SELECT
            ticker,
            COUNT(*) AS row_count,
            SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) AS rsi_nulls,
            SUM(CASE WHEN ma_50 IS NULL THEN 1 ELSE 0 END) AS ma_50_nulls,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM features
        GROUP BY ticker
        ORDER BY ticker
    """).fetchall()

    for row in summary:
        ticker, row_count, rsi_nulls, ma_50_nulls, min_date, max_date = row
        logger.info(
            "%s: %d rows | rsi_14 nulls=%d (~13 expected) | ma_50 nulls=%d (~48 expected) | %s to %s",
            ticker, row_count, rsi_nulls, ma_50_nulls, min_date, max_date,
        )

    logger.info("build_features complete")
    conn.close()


if __name__ == "__main__":
    run_build()