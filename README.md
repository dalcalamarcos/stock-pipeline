# Stock Market Anomaly Detector & Price Predictor

End-to-end MLOps pipeline — daily stock & macro ingestion, feature engineering 
in DuckDB, anomaly detection and direction prediction models, served via FastAPI 
and monitored for data drift with Evidently AI.

## Architecture

> Diagram coming in Phase 5

## Stack

| Layer | Tool |
|---|---|
| Ingestion | yfinance, fredapi |
| Orchestration | Apache Airflow |
| Storage | DuckDB |
| ML | scikit-learn |
| Serving | FastAPI |
| Monitoring | Evidently AI |
| Dashboard | Streamlit |
| Infrastructure | Docker Compose |

## Status

🚧 In progress — Phase 1 (Data Foundation)

## Setup

> Instructions coming once stack is stable