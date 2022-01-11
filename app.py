import os
import pickle
from typing import Literal

import pandas as pd
import yaml
from box import Box
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlp.models.evaluate import inference
from mlp.models.utils import load_model
from mlp.processing.data import process_data

# Project Directories

ROOT = os.getcwd()
CONFIG_FILEPATH = ROOT + "/config.yaml"

# Load config file
with open(CONFIG_FILEPATH, "r", encoding="utf-8") as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

ENCODER_FILEPATH = os.path.join(ROOT, config.data_processing.encoder_filepath)
BINARIZER_FILEPATH = os.path.join(
    ROOT, config.data_processing.binarizer_filepath
)
MODEL_FILEPATH = os.path.join(
    ROOT, config.models.random_forest.output_filepath
)

# The following is needed to enable DVC to pull data on Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class CensusData(BaseModel):
    """
    Copied the attributes from https://archive.ics.uci.edu/ml/datasets/census+income
    """

    age: float
    workclass: Literal[
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ]
    fnlwgt: float
    education: Literal[
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ]
    education_num: float = Field(alias="education-num")
    marital_status: Literal[
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ] = Field(alias="marital-status")
    occupation: Literal[
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ]
    relationship: Literal[
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ]
    race: Literal[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ]
    sex: Literal["Female", "Male"]
    capital_gain: float = Field(alias="capital-gain")
    capital_loss: float = Field(alias="capital-loss")
    hours_per_week: float = Field(alias="hours-per-week")
    native_country: Literal[
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ] = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "fnlwgt": 209642,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States",
            }
        }


app = FastAPI()

# Load objects needed to make prediction
categorical_features = config.app.cat_features
label = config.data_processing.label
encoder = load_model(ENCODER_FILEPATH)
binarizer = load_model(BINARIZER_FILEPATH)
model = load_model(MODEL_FILEPATH, "rb")


@app.get("/")
def root():
    """
    Display welcome greeting
    """
    return {"Greeting": "Welcome!"}


# @app.post('/predict', response_model=EntityOut)
@app.post("/predict")
async def make_predictions(request_data: CensusData):
    request_df = pd.DataFrame(
        {k: v for k, v in request_data.dict().items()}, index=[0]
    )

    X, *_ = process_data(
        X=request_df,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=binarizer,
    )

    preds = inference(model=model, X=X)
    # For some reason we need to add .tolist()[0] to the end for
    # this to work. Here I foolowed this article:
    # https://nickc1.github.io/api,/scikit-learn/2019/01/10/scikit-fastapi.html
    preds = binarizer.inverse_transform(preds).tolist()[0]
    return {"predictions": preds}
