from dataclasses import dataclass
from typing import List


@dataclass()
class FeatureParams:
    numerical_features: List[str]
    categorical_features: List[str]
    target_col: str
