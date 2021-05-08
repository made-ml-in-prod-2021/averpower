ml_project
==============================

Example of ml project

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
 Train mode:

python main_pipeline.py train ../configs/train_config_1.yaml

 Prediction mode:

python main_pipeline.py predict ../configs/predict_config.yaml
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── predictions    <- Predictions made by models.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── configs            <- Configuration files for train mode, prediction mode and logging. 
    |
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── enities        <- parameter enities
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    │   ├── logs           <- log files describing work stages
    │   │
    └── main_pipeline.py   <- main .py file
