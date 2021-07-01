from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from constants import default_args, BASE_DIR

with DAG(
        "dag_download",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(7),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[f"{BASE_DIR}:/data"]
    )

