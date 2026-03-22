import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/opt/airflow/data/stock_pipeline.duckdb")

FEATURE_COLS = [
    'return_1d', 'ma_5', 'ma_20', 'ma_50', 'volatility_20',
    'rsi_14', 'volume_zscore', 'fedfunds', 'cpiaucsl', 'vix', 't10y2y',
]

TRAIN_CUTOFF = '2025-03-20'

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'


def run_training():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DUCKDB_PATH)
    cols = ', '.join(['ticker', 'date'] + FEATURE_COLS + ['target'])
    df = conn.execute(
        f"SELECT {cols} FROM features WHERE target IS NOT NULL ORDER BY ticker, date"
    ).df()
    conn.close()

    df = df.dropna(subset=FEATURE_COLS + ['target'])
    df = df.sort_values(['ticker', 'date'])
    logger.info("Rows after filtering nulls: %d", len(df))

    train_df = df[df['date'].astype(str) < TRAIN_CUTOFF]
    test_df = df[df['date'].astype(str) >= TRAIN_CUTOFF]
    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['target'].values
    X_test = test_df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train_scaled)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    joblib.dump(iso, ARTIFACTS_DIR / 'isolation_forest.pkl')
    joblib.dump(lr, ARTIFACTS_DIR / 'logistic_regression.pkl')
    joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.pkl')

    metadata = {
        'train_date': datetime.now(timezone.utc).isoformat(),
        'train_cutoff_date': TRAIN_CUTOFF,
        'feature_cols': FEATURE_COLS,
        'n_train_rows': len(train_df),
        'n_test_rows': len(test_df),
        'lr_accuracy': None,
        'lr_precision': None,
        'lr_recall': None,
        'lr_f1': None,
        'baseline_accuracy': None,
        'iso_anomaly_rate': None,
        'majority_class': int(np.bincount(y_train.astype(int)).argmax()),
    }
    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete. Models serialized to models/artifacts/")


if __name__ == '__main__':
    run_training()
