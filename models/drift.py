import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

logger = logging.getLogger(__name__)

DUCKDB_PATH = Path(os.environ.get("DUCKDB_PATH", "data/stock_pipeline.duckdb"))
REPORTS_DIR = Path("reports")
TRAIN_CUTOFF = "2025-03-20"
FEATURE_COLS = [
    'return_1d', 'ma_5', 'ma_20', 'ma_50', 'volatility_20',
    'rsi_14', 'volume_zscore', 'fedfunds', 'cpiaucsl', 'vix', 't10y2y',
]


def run_drift_report() -> dict:
    """Generate an Evidently data drift report comparing training vs
    current feature distributions. Saves an HTML report to reports/ and returns
    a per-feature drift summary dict."""

    Path("reports").mkdir(exist_ok=True)

    cols_sql = ", ".join(FEATURE_COLS)

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        ref_df = con.execute(
            f"SELECT {cols_sql} FROM features WHERE date < '{TRAIN_CUTOFF}' ORDER BY date"
        ).fetchdf()
        ref_df.columns = FEATURE_COLS

        curr_df = con.execute(
            f"SELECT {cols_sql} FROM features WHERE date >= CURRENT_DATE - INTERVAL 30 DAYS ORDER BY date"
        ).fetchdf()
        curr_df.columns = FEATURE_COLS
    finally:
        con.close()

    logger.info("Reference window: %s rows, Current window: %s rows", len(ref_df), len(curr_df))

    column_mapping = ColumnMapping(numerical_features=FEATURE_COLS)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

    report.save_html(str(REPORTS_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d')}.html"))

    result = report.as_dict()
    drift_by_columns = result["metrics"][0]["result"]["drift_by_columns"]

    drifted = 0
    for feature, info in drift_by_columns.items():
        if info["drift_detected"]:
            drifted += 1
            logger.warning("Drift detected in feature '%s': score=%.4f", feature, info["drift_score"])

    logger.info("Drift report complete. %d/%d features drifted.", drifted, len(drift_by_columns))

    return drift_by_columns
