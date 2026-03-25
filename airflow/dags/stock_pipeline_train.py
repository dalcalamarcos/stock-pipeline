import sys
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.insert(0, '/opt/airflow')

from models.train import run_training
from models.evaluate import run_evaluation
from models.drift import run_drift_report

default_args = {
    "owner": "marcos",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="stock_pipeline_train",
    default_args=default_args,
    description="Weekly model retraining and evaluation",
    schedule_interval="0 22 * * 0",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["modeling", "phase-4"],
):
    task_train = PythonOperator(
        task_id="train_models",
        python_callable=run_training,
    )

    task_evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=run_evaluation,
    )

    task_drift = PythonOperator(
        task_id="generate_drift_report",
        python_callable=run_drift_report,
    )

    task_train >> task_evaluate >> task_drift
