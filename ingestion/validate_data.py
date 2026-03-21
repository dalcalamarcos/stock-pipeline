import duckdb
from datetime import date
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/opt/airflow/data/stock_pipeline.duckdb")
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,SPY,QQQ").split(",")


def run_validation():
    con = duckdb.connect(DUCKDB_PATH)
    errors = []

    # Check 1: Ensure each expected ticker has at least one row in raw_prices.
    # Missing tickers indicate a failed or incomplete ingestion run.
    for ticker in TICKERS:
        count = con.execute(
            "SELECT COUNT(*) FROM raw_prices WHERE ticker = ?", [ticker]
        ).fetchone()[0]
        if count == 0:
            msg = f"No rows found for ticker {ticker}"
            errors.append(msg)
            logger.error(msg)
        else:
            logger.info("Ticker %s: %d rows found", ticker, count)

    # Check 2: Flag any rows where adj_close is NULL, since downstream calculations
    # depend on adjusted close prices being present for every record.
    rows = con.execute(
        "SELECT ticker, COUNT(*) FROM raw_prices WHERE adj_close IS NULL GROUP BY ticker"
    ).fetchall()
    for ticker, null_count in rows:
        msg = f"Ticker {ticker} has {null_count} row(s) with NULL adj_close"
        errors.append(msg)
        logger.error(msg)

    # Check 3: Confirm data is recent enough to be useful. Stale data (>5 days old)
    # may indicate a broken schedule or source outage.
    for ticker in TICKERS:
        result = con.execute(
            "SELECT MAX(date) FROM raw_prices WHERE ticker = ?", [ticker]
        ).fetchone()[0]
        if result is not None:
            days_old = (date.today() - result).days
            if days_old > 5:
                msg = (
                    f"Ticker {ticker} data is stale: latest date is {result} "
                    f"({days_old} days ago)"
                )
                errors.append(msg)
                logger.warning(msg)

    con.close()

    if errors:
        raise Exception("\n".join(errors))

    logger.info("All validation checks passed.")


if __name__ == "__main__":
    run_validation()
