from datetime import timedelta


BASE_DIR = "/home/mo/PycharmProjects/averpower_3/airflow_ml_dags/data"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}