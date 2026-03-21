import os
import logging
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/opt/airflow/data/stock_pipeline.duckdb")
LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", 5))

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI - Inflation",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "T10Y2Y": "10Y-2Y Treasury Yield Spread",
}


def get_latest_date(conn, series_id):
    """Return the most recent date loaded for a series, or a fallback date for initial load."""
    result = conn.execute(
        "SELECT MAX(date) FROM raw_macro WHERE series_id = ?", [series_id]
    ).fetchone()[0]
    if result is None:
        return date.today() - timedelta(days=365 * LOOKBACK_YEARS)
    return result


def extract_series(conn, fred, series_id, description):
    """Download observations for a single FRED series and insert new rows into raw_macro."""
    started_at = datetime.now()
    last_date = get_latest_date(conn, series_id)
    start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        logger.info("Extracting %s (%s) from %s", series_id, description, start_str)
        series = fred.get_series(series_id, observation_start=start_str)

        if series.empty:
            logger.info("No new data for %s", series_id)
            return {"series_id": series_id, "rows": 0, "status": "success", "started_at": started_at}

        df = series.reset_index()
        df.columns = ["date", "value"]
        df["series_id"] = series_id
        df["date"] = df["date"].dt.date
        df = df.dropna(subset=["value"])
        df = df[["series_id", "date", "value"]]

        conn.execute(
            "INSERT INTO raw_macro (series_id, date, value) "
            "SELECT series_id, date, value FROM df "
            "ON CONFLICT (series_id, date) DO NOTHING"
        )
        rows = len(df)
        logger.info("Loaded %d rows for %s", rows, series_id)
        return {"series_id": series_id, "rows": rows, "status": "success", "started_at": started_at}

    except Exception as exc:
        logger.error("Failed to load %s: %s", series_id, exc)
        return {"series_id": series_id, "rows": 0, "status": "failed", "error": str(exc), "started_at": started_at}


def run_extraction():
    """Orchestrate incremental extraction for all configured FRED series and log each run."""
    fred = Fred(api_key=FRED_API_KEY)
    conn = duckdb.connect(DUCKDB_PATH)

    # Ensure tables exist before loading any data
    with open("/opt/airflow/ingestion/schema.sql") as f:
        conn.execute(f.read())

    results = []
    for series_id, description in FRED_SERIES.items():
        result = extract_series(conn, fred, series_id, description)
        results.append(result)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in results:
        conn.execute(
            """
            INSERT INTO ingestion_log
                (run_id, source, ticker, rows_loaded, status, error_msg, started_at, finished_at)
            VALUES (?, 'fred', ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                r.get("series_id"),
                r.get("rows", 0),
                r.get("status"),
                r.get("error"),
                r.get("started_at"),
                datetime.now(),
            ],
        )

    conn.close()

    failed = [r["series_id"] for r in results if r["status"] == "failed"]
    if failed:
        raise Exception(f"Extraction failed for series: {', '.join(failed)}")


if __name__ == "__main__":
    run_extraction()
