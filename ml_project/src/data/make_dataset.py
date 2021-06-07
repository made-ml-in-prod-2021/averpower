from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
from enities.split_params import SplittingParams


def read_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data


def split_train_val_data(data: pd.DataFrame, params: SplittingParams, target_col: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    stratify = data[target_col] if params.stratify else None
    train_data, val_data = train_test_split(data, test_size=params.val_size, random_state=params.random_state,
                                            stratify=stratify)
    return train_data, val_data
