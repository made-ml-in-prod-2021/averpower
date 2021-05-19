import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import *

SklearnClassificationModel = Union[LogisticRegression, GradientBoostingClassifier]
model: Optional[SklearnClassificationModel] = None
transformer: Optional[Pipeline] = None
columns_info: pd.Series


@app.on_event("startup")
def start():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    model = load_model(model_path)
    global transformer
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    transformer = load_transformer(transformer_path)
    global columns_info
    df = pd.read_csv("data.csv")
    df = df.drop(columns=['target'])
    columns_info = df.dtypes


def test_model_is_set():
    with TestClient(app) as client:
        response = client.get("/healz")
        assert response.status_code == 200


def test_good_request():
    df = pd.read_csv("data.csv")
    sample = df.sample()
    answer = sample["target"]
    sample = sample.drop(columns=["target"])
    with TestClient(app) as client:
        columns = sample.columns.tolist()
        response = client.get("/predict/",
                              json={"data": [sample.values.ravel().tolist()],
                                    "features": columns},
                              )

        assert response.status_code == 200
        assert response.json() == answer.values[0]


@pytest.fixture()
def sample():
    df = pd.read_csv("data.csv")
    sample = df.sample()
    sample.drop(columns=["target"], inplace=True)
    return sample


def test_predict_wrong_column_order(sample):
    with TestClient(app) as client:
        columns = sample.columns.tolist()
        np.random.shuffle(columns)

        response = client.get("/predict/",
                              json={"data": [sample.values.ravel().tolist()],
                                    "features": columns},
                              )
        assert response.status_code == 400
        assert response.json() == {"detail": f"Wrong order of features. Use these order: {sample.columns.tolist()}"}


def test_predict_wrong_columns(sample):
    with TestClient(app) as client:
        columns = sample.columns.tolist()
        columns = np.random.choice(columns, len(columns))

        response = client.get("/predict/",
                              json={"data": [sample.values.ravel().tolist()],
                                    "features": columns.tolist()},
                              )
        assert response.status_code == 400
        assert response.json() == {"detail": f"Wrong set of features. Use these ones: {sample.columns.tolist()}"}


def test_predict_wrong_type(sample):
    with TestClient(app) as client:
        columns = sample.columns.tolist()
        data = sample.values.ravel().tolist()
        data[0] = "wrong"

        response = client.get("/predict/",
                              json={"data": [data],
                                    "features": columns},
                              )
        assert response.status_code == 400
        assert response.json() == {"detail": f"Wrong type for feature {columns[0]}"}
