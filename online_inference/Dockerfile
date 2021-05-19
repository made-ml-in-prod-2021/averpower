FROM python:3.9.4
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY model.pkl ./model.pkl
COPY transformer.pkl ./transformer.pkl
COPY columns.pkl ./columns.pkl
COPY app.py ./app.py

WORKDIR .

ENV PATH_TO_MODEL="./model.pkl"
ENV PATH_TO_TRANSFORMER="./transformer.pkl"
ENV PATH_TO_COL_NAMES="./columns.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]