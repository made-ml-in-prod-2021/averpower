from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from enities.predict_path_params import PredictPathParams


@dataclass()
class PredictParams:
    pathes: PredictPathParams
    test_data_size: int
    target_col: str


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(config_path: str) -> PredictParams:
    with open(config_path, "r") as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
