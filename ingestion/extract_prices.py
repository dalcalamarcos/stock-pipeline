import os
import time
import logging
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/opt/airflow/data/stock_pipeline.duckdb")
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,SPY,QQQ").split(",")
LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", 5))


def get_latest_date(conn, ticker):
    """Return the most recent date loaded for a ticker, or a fallback date for initial load."""
    result = conn.execute(
        "SELECT MAX(date) FROM raw_prices WHERE ticker = ?", [ticker]
    ).fetchone()[0]
    if result is None:
        return date.today() - timedelta(days=365 * LOOKBACK_YEARS)
    return result


def extract_ticker(conn, ticker):
    """Download daily OHLCV data for a single ticker from yfinance and insert new rows."""
    started_at = datetime.now()
    last_date = get_latest_date(conn, ticker)
    start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_str = date.today().strftime("%Y-%m-%d")

    logger.info("Extracting %s from %s to %s", ticker, start_str, end_str)
    df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=False)

    if df.empty:
        logger.info("No new data for %s", ticker)
        return {"ticker": ticker, "rows": 0, "status": "success", "started_at": started_at}

    try:
        # Flatten MultiIndex columns produced by yfinance when downloading a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index(drop=False).reset_index(drop=True)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        df["ticker"] = ticker
        df["date"] = df["date"].dt.date

        df = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
        df = df.dropna(subset=["adj_close"])

        conn.execute(
            "INSERT INTO raw_prices (ticker, date, open, high, low, close, adj_close, volume) "
            "SELECT ticker, date, open, high, low, close, adj_close, volume FROM df "
            "ON CONFLICT (ticker, date) DO NOTHING"
        )
        rows = len(df)
        logger.info("Loaded %d rows for %s", rows, ticker)
        return {"ticker": ticker, "rows": rows, "status": "success", "started_at": started_at}

    except Exception as exc:
        logger.error("Failed to load %s: %s", ticker, exc)
        return {"ticker": ticker, "rows": 0, "status": "failed", "error": str(exc), "started_at": started_at}


def run_extraction():
    """Orchestrate incremental extraction for all configured tickers and log each run."""
    conn = duckdb.connect(DUCKDB_PATH)

    # Ensure tables exist before loading any data
    with open("/opt/airflow/ingestion/schema.sql") as f:
        conn.execute(f.read())

    results = []
    for ticker in TICKERS:
        result = extract_ticker(conn, ticker)
        results.append(result)
        time.sleep(0.5)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in results:
        conn.execute(
            """
            INSERT INTO ingestion_log
                (run_id, source, ticker, rows_loaded, status, error_msg, started_at, finished_at)
            VALUES (?, 'yfinance', ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                r.get("ticker"),
                r.get("rows", 0),
                r.get("status"),
                r.get("error"),
                r.get("started_at"),
                datetime.now(),
            ],
        )

    conn.close()

    failed = [r["ticker"] for r in results if r["status"] == "failed"]
    if failed:
        raise Exception(f"Extraction failed for tickers: {', '.join(failed)}")


if __name__ == "__main__":
    run_extraction()
