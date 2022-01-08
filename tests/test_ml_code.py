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

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.processing.data import clean_data, process_data

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
if __name__ == "__main__":
    test_clean_data()