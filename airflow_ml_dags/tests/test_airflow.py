import pytest
from airflow.models import DagBag


@pytest.fixture()
def dags():
    return DagBag(dag_folder="../dags", include_examples=False)


def test_dag_bag_loaded_correct(dags):
    assert dags.dags is not None
    assert dags.import_errors == {}


def test_first_dag_loaded(dags):
    assert "dag_1" in dags.dags
    assert len(dags.dags["dag_1"].tasks) == 1


def test_second_dag_loaded(dags):
    assert "dag_2" in dags.dags
    assert len(dags.dags["dag_2"].tasks) == 5


def test_third_dag_loaded(dags):
    assert "dag_3" in dags.dags
    assert len(dags.dags["dag_3"].tasks) == 2


def test_second_dag_content(dags):
    content = {
        "airflow-wait-file": ["docker-airflow-preprocess"],
        "docker-airflow-preprocess": ["docker-airflow-split"],
        "docker-airflow-split": ['docker-airflow-train'],
        "docker-airflow-train": ["docker-airflow-validate"],
        "docker-airflow-validate": []
    }
    dag = dags.dags["dag_2"]
    for name, task in dag.task_dict.items():
        assert set(content[name]) == task.downstream_task_ids


def test_first_dag_content(dags):
    content = {
        "docker-airflow-download": [],
    }
    dag = dags.dags["dag_1"]
    for name, task in dag.task_dict.items():
        assert set(content[name]) == task.downstream_task_ids


def test_third_dag_content(dags):
    content = {
        "airflow-wait-file": ["docker-airflow-predict"],
        "docker-airflow-predict": []
    }
    dag = dags.dags["dag_3"]
    for name, task in dag.task_dict.items():
        assert set(content[name]) == task.downstream_task_ids
