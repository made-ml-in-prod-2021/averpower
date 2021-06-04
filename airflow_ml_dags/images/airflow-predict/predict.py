import os
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    model = pickle.load(open(os.path.join(model_dir, "model.pkl"), "rb"))

    drop_cols = ["sex", "cp", "restecg", "exang", "slope", "ca", "thal"]
    data.drop(columns=drop_cols, inplace=True)

    poly_features = PolynomialFeatures(degree=2)
    new_features = poly_features.fit_transform(data)
    prediction = model.predict(new_features)
    os.makedirs(output_dir, exist_ok=True)
    prediction_df = pd.DataFrame(prediction)
    prediction_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()