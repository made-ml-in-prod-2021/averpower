FROM airflow-ml-base:latest

COPY download.py /download.py
COPY data_base.csv /data_base.csv

ENV DATA_BASE_PATH="data_base.csv"

ENTRYPOINT ["python", "download.py"]