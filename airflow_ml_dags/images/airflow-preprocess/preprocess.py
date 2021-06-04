import os
import pandas as pd
import click
import pickle
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


@click.group()
def main():
    pass


@main.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    drop_cols = ["sex", "cp", "restecg", "exang", "slope", "ca", "thal"]
    data.drop(columns=drop_cols, inplace=True)

    poly_features = PolynomialFeatures(degree=2)
    new_features = poly_features.fit_transform(data)

    df_processed = pd.DataFrame(data=new_features)
    df_processed["target"] = target["target"]

    os.makedirs(output_dir, exist_ok=True)
    df_processed.to_csv(os.path.join(output_dir, "data.csv"), index=False)


@main.command("split")
@click.option("--input-dir")
def split(input_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = data["target"]
    data = data.drop(columns=["target"])
    X_train, X_val, y_train, y_val = train_test_split(data, target)
    df_train = pd.DataFrame(X_train)
    df_train["target"] = y_train
    df_val = pd.DataFrame(X_val)
    df_val["target"] = y_val
    df_train.to_csv(os.path.join(input_dir, "data_train.csv"), index=False)
    df_val.to_csv(os.path.join(input_dir, "data_val.csv"), index=False)


@main.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir):
    data_train = pd.read_csv(os.path.join(input_dir, "data_train.csv"))
    y_train = data_train["target"]
    X_train = data_train.drop(columns=["target"])
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(log_reg, f)


@main.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
def validate(data_dir: str, model_dir):
    model = pickle.load(open(os.path.join(model_dir, "model.pkl"), "rb"))
    data_val = pd.read_csv(os.path.join(data_dir, "data_val.csv"))
    y_val = data_val["target"]
    X_val = data_val.drop(columns=["target"])
    prediction = model.predict(X_val)

    metrics_dict = {
        "roc_auc_score": roc_auc_score(y_val, prediction),
        "precision": precision_score(y_val, prediction),
        "recall": recall_score(y_val, prediction),
        "f1_score": f1_score(y_val, prediction),
    }

    with open(os.path.join(model_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics_dict, metric_file)


if __name__ == '__main__':
    main()