import os
from datetime import timedelta, date
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor
import logging


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_file():
    return os.path.exists("/opt/airflow/data")


with DAG(
        "dag_2",
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
        volumes=["/home/mo/PycharmProjects/averpower_3/airflow_ml_dags/data:/data"]
    )

    split = DockerOperator(
        image="airflow-preprocess",
        command="split --input-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=["/home/mo/PycharmProjects/averpower_3/airflow_ml_dags/data:/data"]
    )

    train = DockerOperator(
        image="airflow-preprocess",
        command="train --input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=["/home/mo/PycharmProjects/averpower_3/airflow_ml_dags/data:/data"]
    )

    validate = DockerOperator(
        image="airflow-preprocess",
        command="validate --data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=["/home/mo/PycharmProjects/averpower_3/airflow_ml_dags/data:/data"]
    )

    wait >> preprocess >> split >> train >> validate
