import pickle
import json
from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from enities.gridsearch_params import GridSearchParams

SklearnClassificationModel = Union[LogisticRegression, GradientBoostingClassifier]


def train_model(features: pd.DataFrame,
                target: pd.Series, model_type: str, gs_params: GridSearchParams) -> SklearnClassificationModel:
    if model_type == "LogisticRegression":
        model = GridSearchCV(LogisticRegression(), gs_params.param_grid, cv=gs_params.cv, scoring=gs_params.scoring)
    elif model_type == "GradientBoostingClassifier":
        model = GridSearchCV(GradientBoostingClassifier(), gs_params.param_grid, cv=gs_params.cv,
                             scoring=gs_params.scoring)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model.best_estimator_


def predict_model(model: SklearnClassificationModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def save_metrics(metric_path: str, metrics: Dict[str, float]):
    with open(metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


def save_model(model: SklearnClassificationModel, output: str):
    with open(output, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path: str) -> SklearnClassificationModel:
    return pickle.load(open(model_path, 'rb'))
