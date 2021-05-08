from dataclasses import dataclass
from enities.split_params import SplittingParams
from enities.gridsearch_params import GridSearchParams
from enities.feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainParams:
    data_path: str
    model_path: str
    transformer_path: str
    metric_path: str
    log_path: str
    splitting_params: SplittingParams
    model_type: str
    feature_params: FeatureParams
    gs_params: GridSearchParams


TrainingParamsSchema = class_schema(TrainParams)


def read_training_params(config_path: str) -> TrainParams:
    with open(config_path, "r") as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
