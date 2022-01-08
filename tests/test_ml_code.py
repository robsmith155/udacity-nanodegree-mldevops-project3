"""
This module contains tests for the functions contained withing the mlp package.
This includes the data processing and model training code.

@author: Rob Smith
date: 8th January 2022
"""

import logging
import yaml
from box import Box
import sys
from pathlib import Path
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import numpy as np
import imblearn

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.processing.data import clean_data, process_data
from mlp.models.random_forest import train_model

# Project Directories
PACKAGE_ROOT = Path(mlp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILEPATH = ROOT / "config.yaml"

# Load config file
with open(CONFIG_FILEPATH, "r", encoding='utf-8') as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

# Set up logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

LOG_FILEPATH = ROOT / '.logs/test_mlp.log'
logging.basicConfig(
    filename = LOG_FILEPATH,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

RAW_DATA_FILEPATH = ROOT / config.data.raw.filepath
CLEANED_DATA_FILEPATH = ROOT / config.data.cleaned.filepath
TRAINING_DATA_FILEPATH = ROOT / config.data_processing.train_filepath


@pytest.fixture(scope="session")
def df_clean():
    df = pd.read_csv(CLEANED_DATA_FILEPATH)
    return df

@pytest.fixture(scope="session")
def training_x():
    with open(TRAINING_DATA_FILEPATH, 'rb') as f:
        X_train = np.load(f)
    return X_train

@pytest.fixture(scope="session")
def training_y():
    with open(TRAINING_DATA_FILEPATH, 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)
    return y_train

def test_clean_data() -> None:
    """
    Test that the clean_data() function cleans the data properly 
    """
    df_clean = clean_data(filepath=RAW_DATA_FILEPATH)

    # Make sure that no missing data
    try:
        assert df_clean.isnull().sum().sum() == 0
        logging.info('Testing clean_data: No rows with null values.')
    except AssertionError as err:
        logging.error(f'Testing clean_data: Expected no null values but found {df_clean.isnull().sum().sum()}')
        raise err

    # Make sure that the number of rows is reasonable 
    try:
        assert len(df_clean) > 25000
        assert len(df_clean) < 35000
        logging.info(f'Testing clean data: The cleaned dataset has {len(df_clean)} rows which is within the expected range.')
    except AssertionError as err:
        logging.error(f'Testing clean_data: After cleaning the dataset has {len(df_clean)} which is outside the expected range.')
        raise err

    # Ensure the number of columns is unchanged
    try:
        assert len(df_clean.columns) == 15
        logging.info(f'Testing clean_data: The cleaned dataset has the correct number of columns ({len(df_clean.columns)})') 
    except AssertionError as err:
        logging.error(f'Testing clean_data: The cleaned dataset has {len(df_clean.columns)} but expected 15.')
        raise err


def test_process_data_training(df_clean) -> None:
    """
    Test that the process_data function works as expected
    """
    train, _ = train_test_split(df_clean, test_size=0.20)
    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        X=train, 
        categorical_features=config.data_processing.cat_features,
        label=config.data_processing.label,
        training=True
    )

    # Check that the output data are arrays
    try:
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        logging.info('Testing process_data: Output data are Numpy arrays')
    except AssertionError as err:
        logging.error(f'Testing process_data: Expected output data to be Numpy arrays but they are {type(X_train)} and {type(y_train)}')

    # Check that y is rank 1
    try:
        assert y_train.ndim == 1
        logging.info('Testing process_data: y_train has rank of 1')
    except AssertionError as err:
        logging.error('Testing process_data: Expected rank of y_train to be 1, but got {y_train.dim}.')
        raise err


def test_train_model(training_x, training_y) -> None:
    """
    Test that the train_model function performs as expected
    """

    model = train_model(training_x, training_y, config)

    # Verify the model type
    try:
        assert isinstance(model, imblearn.ensemble._forest.BalancedRandomForestClassifier)
        logging.info('Testing train_model: The model is the correct type.')
    except AssertionError as err:
        logging.error(f'Testing train_model: Expected the model to be of the type BalancedRandomForestClassifier but got {type(model)}.')
        raise err

    # Verify that the model has been fitted
    try:
        model.predict(training_x)
        logging.info('Testing train_model: Model has been fitted.')
    except NotFittedError as err:
        logging.error('Testing train_model: The model has not been fitted.')
        raise err


if __name__ == "__main__":
    test_clean_data()