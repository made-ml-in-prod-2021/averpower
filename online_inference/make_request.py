import pandas as pd
import numpy as np
import click
from faker import Faker
from faker.providers import BaseProvider
import requests


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
                m = df[column].mean()
                s = df[column].std()
                person.append(np.random.normal(m, s))
            else:
                uni = df[column].unique()
                person.append(np.random.choice(uni))

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


@click.command()
@click.argument("resp_num", default=10, type=int)
def request_generation(resp_num):
    base_data = read_data_without_target('data.csv', 'target')
    fake_data = generate_fake_data(base_data, resp_num)
    request_features = list(fake_data.columns)
    for i in range(resp_num):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in fake_data.iloc[i].tolist()
        ]
        response = requests.get(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())


if __name__ == "__main__":
    request_generation()
