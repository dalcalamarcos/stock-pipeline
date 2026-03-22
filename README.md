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

Completed — Phase 1 (Data Foundation)
Completed — Phase 2 (Feature Engineering)
In progress — Phase 3 (Modeling)

## Setup

> Instructions coming once stack is stable