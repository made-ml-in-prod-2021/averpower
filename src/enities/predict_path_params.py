from dataclasses import dataclass, field


@dataclass()
class PredictPathParams:
    model_path: str
    transformer_path: str
    data_source_path: str
    output_path: str
    log_path: str = field(default="../configs/logging_config.yaml")
