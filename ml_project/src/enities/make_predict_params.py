from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictParams:
    model_path: str
    transformer_path: str
    data_predict_path: str
    output_path: str
    log_path: str = field(default="../configs/logging_config.yaml")


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(config_path: str) -> PredictParams:
    with open(config_path, "r") as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
