import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from enities.feature_params import FeatureParams


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    return numerical_pipeline


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_transformer(params: FeatureParams) -> Pipeline:
    transformer = Pipeline(
        ["col_transformer", ColumnTransformer(
            [
                ("numerical_pipeline", build_numerical_pipeline(), params.numerical_features),
                ("categorical_features", build_categorical_pipeline(), params.categorical_features),
            ]
        ),
         ("poly", PolynomialFeatures(degree=2)),
         ]
    )

    return transformer


def save_transformer(transformer: Pipeline, output: str):
    with open(output, "wb") as f:
        pickle.dump(transformer, f)


def load_transformer(transformer_path: str) -> Pipeline:
    return pickle.load(open(transformer_path, 'rb'))
