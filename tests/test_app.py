"""
This module is to test the FastAPI app in the app.py file in the 
project root.

@author: Rob Smith
Date: 10th Jan 2022
"""

import json
import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from app import app

PACKAGE_ROOT = Path(mlp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

# Set up logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

LOG_FILEPATH = ROOT / ".logs/test_app.log"
logging.basicConfig(
    filename=LOG_FILEPATH,
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

# Setup Pytest fixtures containing test data
@pytest.fixture(scope="session")
def below_50k_sample():
    input_dict = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    return input_dict


@pytest.fixture(scope="session")
def above_50k_sample():
    input_dict = {
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
    return input_dict


# Instantiate the testing client with our app
client = TestClient(app)


def test_get_root() -> None:
    """
    Test the GET method of the app
    """
    r = client.get("/")
    try:
        assert r.status_code == 200
        logging.info("Testing root GET method: Root of app returns 200 status")
    except AssertionError as err:
        logging.error(
            f"Testing root GET method: Expected response status code of 200 but got {r.status_code}"
        )
        raise err

    try:
        assert r.json() == {"Greeting": "Welcome!"}
        logging.info(
            f"Testing root GET method: Returns correct response of {r.json()}"
        )
    except:
        logging.error(
            f"Testing root GET method: Expected response body of Greeting:Welcome! but returned {r.json()}."
        )


def test_below_50k_sample(below_50k_sample) -> None:
    """
    Test the POST method of the FastAPI app with a sample expected to return a
    negative (i.e. earns <$50k) result
    """
    r = client.post("/predict", data=json.dumps(below_50k_sample))
    try:
        assert r.status_code == 200
        logging.info("Testing POST method: Request returns 200 status")
    except AssertionError as err:
        logging.error(
            f"Testing POST method: Expected response status code of 200 but got {r.status_code}"
        )
        raise err

    try:
        assert r.json().get("predictions") == "<=50K"
        logging.info(
            "Testing POST <50k sample: Request returned correct prediction"
        )
    except AssertionError as err:
        logging.error(
            f"Testing POST <50k sample: Expected model to predict <50k but returned"
        )
        raise err


def test_above_50k_sample(above_50k_sample) -> None:
    """
    Test the POST method of the FastAPI with a sample expected to return a
    positive (i.e. earns >$50k) result
    """
    r = client.post("/predict", data=json.dumps(above_50k_sample))
    try:
        assert r.status_code == 200
        logging.info("Testing POST method: Request returns 200 status")
    except AssertionError as err:
        logging.error(
            f"Testing POST method: Expected response status code of 200 but got {r.status_code}"
        )
        raise err

    try:
        assert r.json().get("predictions") == ">50K"
        logging.info(
            "Testing POST >50k sample: Request returned correct prediction"
        )
    except AssertionError as err:
        logging.error(
            f"Testing POST >50k sample: Expected model to predict >50k but returned"
        )
        raise err
