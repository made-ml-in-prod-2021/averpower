from data.make_dataset import read_data, split_train_val_data
from data.fake_dataset import read_data_without_target, generate_fake_data
from features.make_features import build_transformer, save_transformer, load_transformer
from enities.make_train_params import TrainParams, read_training_params
from enities.make_predict_params import PredictParams, read_predict_params
from models.model_fit_predict import train_model, predict_model, evaluate_model, save_model, save_metrics, load_model
import click
import pandas as pd
import yaml
import logging
import logging.config

APPLICATION_NAME = "ml_project"
logger = logging.getLogger(APPLICATION_NAME)


def setup_logging(logging_config_path: str):
    with open(logging_config_path) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def process_train_model(params: TrainParams):
    data = read_data(params.data_path)
    logger.info(f"Read data. It has {data.shape} shape")
    data_train, data_val = split_train_val_data(data, params.splitting_params, params.feature_params.target_col)
    logger.info(f"Split data. Train data has {data_train.shape} shape, validation data has {data_val.shape} shape")

    transformer = build_transformer(params.feature_params)
    save_transformer(transformer, params.transformer_path)
    logger.info(f"Saved transformer on path {params.transformer_path}")
    transformer.fit(data_train)
    transformed_features_train = pd.DataFrame(transformer.transform(data_train))
    transformed_features_val = pd.DataFrame(transformer.transform(data_val))
    logger.info(
        f"Transformed data. Train data has {transformed_features_train.shape} shape, validation data has {transformed_features_val.shape} shape")

    target_train = data_train[params.feature_params.target_col]
    target_val = data_val[params.feature_params.target_col]

    model = train_model(transformed_features_train, target_train, params.model_type, params.gs_params)
    logger.info(f"Learned {params.model_type} model with these params:\n{model.best_params_}")
    save_model(model, params.model_path)
    logger.info(f"Saved model to path {params.model_path}")
    predicts = predict_model(model, transformed_features_val)
    logger.info(f"Prediction for model {params.model_type} was made")
    metrics = evaluate_model(predicts, target_val)
    logger.info(f"Metrics for prediction are:\n {metrics}")
    save_metrics(params.metric_path, metrics)


def process_predict(params: PredictParams):
    data = read_data_without_target(params.pathes.data_source_path, params.target_col)
    logger.info(f"Read source data. It has {data.shape} shape")
    test_data = generate_fake_data(data, params.test_data_size)
    logger.info(f"Generated fake test data. It has {test_data.shape} shape")

    model = load_model(params.pathes.model_path)
    transformer = load_transformer(params.pathes.transformer_path)
    transformed_features = pd.DataFrame(transformer.fit_transform(test_data))
    predictions = model.predict(transformed_features)
    logger.info(f"Prediction of {predictions.shape} shape was made")

    with open(params.pathes.output_path, 'w') as output_stream:
        output_stream.writelines(list(map(lambda x: str(x) + "\n", predictions)))
    logger.info(f"Predictions were saved on path {params.pathes.output_path}")


@click.group()
def main():
    pass


@main.command("train")
@click.argument("config_path", default="../configs/train_config_1.yaml")
def train_pipeline_command(config_path: str):
    params = read_training_params(config_path)
    setup_logging(params.log_path)
    process_train_model(params)


@main.command("predict")
@click.argument("config_path", default="../configs/predict_config.yaml")
def predict_pipeline_command(config_path: str):
    params = read_predict_params(config_path)
    setup_logging(params.pathes.log_path)
    process_predict(params)


if __name__ == "__main__":
    main()
