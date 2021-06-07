import os
from datetime import date
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor
from constants import default_args, BASE_DIR


def _wait_for_file():
    data_exists = os.path.exists(os.path.join("/opt/airflow/data/raw", date.today().strftime("%Y-%m-%d"), "data.csv"))
    target_exists = os.path.exists(
        os.path.join("/opt/airflow/data/raw", date.today().strftime("%Y-%m-%d"), "target.csv"))
    return data_exists & target_exists


today = date.today().strftime("%Y-%m-%d")
Variable.set("model", f"/data/models/{today}")

with DAG(
        "dag_train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(0),
) as dag:
    wait = PythonSensor(
        task_id="airflow-wait-file",
        python_callable=_wait_for_file,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="preprocess --input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"{BASE_DIR}:/data"]
    )

    split = DockerOperator(
        image="airflow-preprocess",
        command="split --input-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"{BASE_DIR}:/data"]
    )

    train = DockerOperator(
        image="airflow-preprocess",
        command="train --input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{BASE_DIR}:/data"]
    )

    validate = DockerOperator(
        image="airflow-preprocess",
        command="validate --data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"{BASE_DIR}:/data"]
    )

    wait >> preprocess >> split >> train >> validate
