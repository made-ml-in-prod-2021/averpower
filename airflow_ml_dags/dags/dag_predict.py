from datetime import date
import os
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor
from constants import default_args, BASE_DIR


def _wait_for_file():
    return os.path.exists(os.path.join("/opt/airflow/data/raw", date.today().strftime("%Y-%m-%d"), "data.csv"))


with DAG(
        "dag_predict",
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
        volumes=[f"{BASE_DIR}:/data"]
    )
    wait >> predict
