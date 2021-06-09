import os
import pickle
from typing import List, Union, Optional
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

SklearnClassificationModel = Union[LogisticRegression, GradientBoostingClassifier]
model: Optional[SklearnClassificationModel] = None
transformer: Optional[Pipeline] = None
columns_names: List[str]
app = FastAPI()


def load_model(model_path: str) -> SklearnClassificationModel:
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        raise RuntimeError(err)
    return pickle.load(open(model_path, 'rb'))


def load_transformer(transformer_path: str) -> Pipeline:
    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER {transformer_path} is None"
        raise RuntimeError(err)
    return pickle.load(open(transformer_path, 'rb'))


def load_column_names(data_path: str) -> List[str]:
    if data_path is None:
        err = f"PATH_TO_MODEL {data_path} is None"
        raise RuntimeError(err)
    with open(data_path, 'rb') as fin:
        return pickle.load(fin)


class HeartDeceaseModel(BaseModel):
    data: List[conlist(Union[float, str], min_items=13, max_items=13)]
    features: List[str]


def make_predict(data: List, features: List[str], model: SklearnClassificationModel, transformer: Pipeline) -> int:
    data_frame = pd.DataFrame(data, columns=features)
    prediction = model.predict(transformer.transform(data_frame))
    return prediction


def validate_request(request: HeartDeceaseModel, columns_names: List):
    if request.features != columns_names:
        if set(request.features) == set(columns_names):
            raise HTTPException(status_code=400, detail=f"Wrong order of features. Use these order: {columns_names}")
        else:
            raise HTTPException(status_code=400, detail=f"Wrong set of features. Use these ones: {columns_names}")
    for req_val, col_name in zip(request.data[0], columns_names):
        if isinstance(req_val, str):
            raise HTTPException(status_code=400, detail=f"Wrong type for feature {col_name}")


@app.get("/")
def main():
    return "Welcome to our predictor"


@app.on_event("startup")
def start():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    model = load_model(model_path)
    global transformer
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    transformer = load_transformer(transformer_path)
    global columns_names
    data_path = os.getenv("PATH_TO_COL_NAMES")
    columns_names = load_column_names(data_path)


@app.get("/healz")
def health() -> bool:
    return not (model is None and transformer is None)


@app.get("/predict/", response_model=int)
def predict(request: HeartDeceaseModel):
    validate_request(request, columns_names)
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
