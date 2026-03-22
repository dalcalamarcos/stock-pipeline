import json
import logging
import os
from pathlib import Path

import duckdb
import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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


def run_evaluation():
    lr = joblib.load(ARTIFACTS_DIR / 'logistic_regression.pkl')
    iso = joblib.load(ARTIFACTS_DIR / 'isolation_forest.pkl')
    scaler = joblib.load(ARTIFACTS_DIR / 'scaler.pkl')

    with open(ARTIFACTS_DIR / 'metadata.json') as f:
        metadata = json.load(f)

    conn = duckdb.connect(DUCKDB_PATH)
    cols = ', '.join(['ticker', 'date'] + FEATURE_COLS + ['target'])
    df = conn.execute(
        f"SELECT {cols} FROM features WHERE target IS NOT NULL ORDER BY ticker, date"
    ).df()
    conn.close()

    df = df.dropna(subset=FEATURE_COLS + ['target'])
    df = df.sort_values(['ticker', 'date'])

    test_df = df[df['date'].astype(str) >= TRAIN_CUTOFF]

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['target'].values.astype(int)

    X_test_scaled = scaler.transform(X_test)

    y_pred = lr.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, y_pred)
    lr_precision = precision_score(y_test, y_pred, zero_division=0)
    lr_recall = recall_score(y_test, y_pred, zero_division=0)
    lr_f1 = f1_score(y_test, y_pred, zero_division=0)

    majority_class = metadata['majority_class']
    baseline_preds = np.full_like(y_test, majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    feature_weights = dict(zip(FEATURE_COLS, lr.coef_[0]))
    sorted_weights = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("LR feature weights (sorted by absolute magnitude):")
    for feature, weight in sorted_weights:
        logger.info("  %-20s %+.4f", feature, weight)

    logger.info(
        "LR — accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | baseline_accuracy=%.4f",
        lr_accuracy, lr_precision, lr_recall, lr_f1, baseline_accuracy,
    )

    anomaly_labels = iso.predict(X_test_scaled)
    anomaly_rate = (anomaly_labels == -1).mean()
    anomaly_dates = test_df['date'].values[anomaly_labels == -1]
    logger.info("IsolationForest anomaly rate: %.4f | sample dates (first 10): %s",
                anomaly_rate, anomaly_dates[:10].tolist())

    metadata.update({
        'lr_accuracy': round(lr_accuracy, 4),
        'lr_precision': round(lr_precision, 4),
        'lr_recall': round(lr_recall, 4),
        'lr_f1': round(lr_f1, 4),
        'baseline_accuracy': round(baseline_accuracy, 4),
        'iso_anomaly_rate': round(float(anomaly_rate), 4),
    })

    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Evaluation complete. metadata.json updated.")


if __name__ == '__main__':
    run_evaluation()
