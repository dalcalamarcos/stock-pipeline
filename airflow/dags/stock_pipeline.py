import sys
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.insert(0, "/opt/airflow/ingestion")

from extract_prices import run_extraction as extract_prices
from extract_macro import run_extraction as extract_macro
from validate_data import run_validation

default_args = {
    "owner": "marcos",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="stock_pipeline",
    default_args=default_args,
    description="Daily stock price and macro data ingestion",
    schedule_interval="0 21 * * 1-5",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "phase-1"],
):
    task_extract_prices = PythonOperator(
        task_id="extract_prices",
        python_callable=extract_prices,
    )

    task_extract_macro = PythonOperator(
        task_id="extract_macro",
        python_callable=extract_macro,
    )

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=run_validation,
    )

    # Tasks run sequentially due to DuckDB's single-writer file lock constraint.
    task_extract_prices >> task_extract_macro >> task_validate
