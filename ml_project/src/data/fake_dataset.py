import pandas as pd
import numpy as np
from faker import Faker
from faker.providers import BaseProvider


def read_data_without_target(data_path: str, target_col: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data.drop(columns=[target_col], inplace=True)
    return data


class DataProvider(BaseProvider):
    @staticmethod
    def generate_row(df: pd.DataFrame) -> np.array:
        person = list()
        for column in df.columns:
            if len(df[column].unique()) > 6:
                person.append(np.random.normal(df[column].mean(), df[column].std()))
            else:
                person.append(np.random.choice(df[column].unique()))

        return np.array(person)


def generate_fake_data(data: pd.DataFrame, test_data_size: int) -> pd.DataFrame:
    fake = Faker()
    fake.add_provider(DataProvider)
    test_array = []
    for _ in range(test_data_size):
        test_array.append(fake.generate_row(data))
    test_df = pd.DataFrame(data=np.array(test_array), columns=data.columns)

    df_dtypes_dict = {
        col: d_type for col, d_type in zip(data.columns, data.dtypes.values)
    }
    test_df.astype(df_dtypes_dict)
    return test_df
