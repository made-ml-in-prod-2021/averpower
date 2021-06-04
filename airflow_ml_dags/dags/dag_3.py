from datetime import timedelta, date
import os
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_file():
    return os.path.exists("/opt/airflow/data")


with DAG(
        "dag_3",
        default_args=default_args,
        schedule_interval="@daily",
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

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --model-dir {{var.value.model}} --output-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=["/home/mo/PycharmProjects/averpower_3/airflow_ml_dags/data:/data"]
    )
    wait >> predict
