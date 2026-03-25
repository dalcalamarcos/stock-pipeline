import datetime
import logging
import os
import joblib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

DUCKDB_PATH = Path(os.environ.get("DUCKDB_PATH", "data/stock_pipeline.duckdb"))
ARTIFACTS_DIR = Path("models/artifacts")

FEATURE_COLS = [
    'return_1d', 'ma_5', 'ma_20', 'ma_50', 'volatility_20',
    'rsi_14', 'volume_zscore', 'fedfunds', 'cpiaucsl', 'vix', 't10y2y',
]

models: dict = {}


def _load_pickle(name: str, key: str) -> None:
    path = ARTIFACTS_DIR / name
    try:
        models[key] = joblib.load(path)
    except FileNotFoundError:
        logger.error("Artifact not found: %s", path)
        models[key] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_pickle("logistic_regression.pkl", "lr")
    _load_pickle("isolation_forest.pkl", "iso")
    _load_pickle("scaler.pkl", "scaler")

    meta_path = ARTIFACTS_DIR / "metadata.json"
    try:
        with open(meta_path) as f:
            models["metadata"] = json.load(f)
    except FileNotFoundError:
        logger.error("Artifact not found: %s", meta_path)
        models["metadata"] = None

    yield

    models.clear()


app = FastAPI(lifespan=lifespan)


def get_latest_features(ticker: str, conn) -> dict | None:
    row = conn.execute(
        f"""
        SELECT {', '.join(FEATURE_COLS)}, date
        FROM features
        WHERE ticker = ? AND return_1d IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """,
        [ticker],
    ).fetchone()

    if row is None:
        return None

    col_names = FEATURE_COLS + ["date"]
    result = dict(zip(col_names, row))

    feature_date = result["date"]
    if isinstance(feature_date, datetime.datetime):
        feature_date = feature_date.date()
    elif isinstance(feature_date, str):
        feature_date = datetime.date.fromisoformat(feature_date)

    stale = (datetime.date.today() - feature_date).days > 3
    result["stale"] = stale
    result["date"] = feature_date

    return result


@app.get("/health")
def health():
    meta = models.get("metadata") or {}
    all_present = all(models.get(k) is not None for k in ("lr", "iso", "scaler"))

    return {
        "status": "ok" if all_present else "degraded",
        "model_train_date": meta.get("train_date"),
        "train_cutoff_date": meta.get("train_cutoff_date"),
        "lr_accuracy": meta.get("lr_accuracy"),
        "lr_f1": meta.get("lr_f1"),
        "baseline_accuracy": meta.get("baseline_accuracy"),
        "iso_anomaly_rate": meta.get("iso_anomaly_rate"),
        "feature_cols": FEATURE_COLS,
        "artifacts_present": {
            "logistic_regression": models.get("lr") is not None,
            "isolation_forest": models.get("iso") is not None,
            "scaler": models.get("scaler") is not None,
        },
    }


@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        with conn:
            features = get_latest_features(ticker, conn)
    except duckdb.IOException:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if features is None:
        raise HTTPException(status_code=404, detail=f"No feature data found for ticker {ticker}")

    if models.get("lr") is None or models.get("scaler") is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable: logistic_regression or scaler not loaded",
        )

    X = np.array([[features[col] for col in FEATURE_COLS]], dtype=float)
    X_scaled = models["scaler"].transform(X)
    proba = models["lr"].predict_proba(X_scaled)
    probability_up = float(proba[0][1])
    prediction = "up" if probability_up >= 0.5 else "down"

    meta = models.get("metadata") or {}
    feature_date = features["date"]
    if isinstance(feature_date, datetime.date):
        feature_date = feature_date.isoformat()

    return {
        "ticker": ticker,
        "prediction": prediction,
        "probability_up": probability_up,
        "last_feature_date": feature_date,
        "model_train_date": meta.get("train_date"),
        "feature_freshness": "stale" if features["stale"] else "ok",
    }


@app.get("/anomalies/{ticker}")
def anomalies(ticker: str):
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        with conn:
            rows = conn.execute(
                f"""
                SELECT {', '.join(FEATURE_COLS)}, date
                FROM features
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 30
                """,
                [ticker],
            ).fetchall()
    except duckdb.IOException:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not rows:
        raise HTTPException(status_code=404, detail=f"No feature data found for ticker {ticker}")

    if models.get("iso") is None or models.get("scaler") is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable: isolation_forest or scaler not loaded",
        )

    # Rows came DESC; reverse to ascending order
    rows = list(reversed(rows))
    col_names = FEATURE_COLS + ["date"]

    X = np.array([[r[i] for i in range(len(FEATURE_COLS))] for r in rows], dtype=float)
    X_scaled = models["scaler"].transform(X)
    preds = models["iso"].predict(X_scaled)
    is_anomaly_flags = [p == -1 for p in preds]

    anomaly_rate = round(sum(is_anomaly_flags) / len(is_anomaly_flags), 4)

    anomaly_list = []
    for row, flag in zip(rows, is_anomaly_flags):
        date_val = row[len(FEATURE_COLS)]
        if isinstance(date_val, datetime.datetime):
            date_val = date_val.date().isoformat()
        elif isinstance(date_val, datetime.date):
            date_val = date_val.isoformat()
        else:
            date_val = str(date_val)
        anomaly_list.append({"date": date_val, "is_anomaly": bool(flag)})

    return {
        "ticker": ticker,
        "window_days": 30,
        "anomaly_rate": anomaly_rate,
        "anomalies": anomaly_list,
    }
