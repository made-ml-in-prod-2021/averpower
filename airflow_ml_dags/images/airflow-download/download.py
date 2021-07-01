import os
from typing import Tuple

import click
import pandas as pd
import numpy as np


def generate_alike(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample = []
    for col in df.columns[:-1]:
        sample.append(np.random.choice(df[col], 100))
    df_sample = pd.DataFrame(data=np.array(sample).T, columns=df.columns[:-1])
    target = np.random.choice(df["target"], 100)
    df_target = pd.DataFrame(data=target.T, columns=["target"])
    return df_sample, df_target


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    data_path = os.getenv("DATA_BASE_PATH")
    df = pd.read_csv(data_path)
    df_sample, df_target = generate_alike(df)
    os.makedirs(output_dir, exist_ok=True)
    df_sample.to_csv(os.path.join(output_dir, "data.csv"))
    df_target.to_csv(os.path.join(output_dir, "target.csv"))


if __name__ == '__main__':
    download()
