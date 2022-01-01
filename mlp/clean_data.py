import pandas as pd
import yaml
from box import Box
import sys
from pathlib import Path

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.processing.data import clean_data

# Project Directories
PACKAGE_ROOT = Path(mlp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILEPATH = ROOT / "config.yaml"

# Load config file
with open(CONFIG_FILEPATH, "r", encoding='utf-8') as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

RAW_DATA_FILEPATH = ROOT / config.data.raw.filepath
CLEANED_DATA_FILEPATH = ROOT / config.data.cleaned.filepath

def run_data_cleaning() -> None:
    df_clean = clean_data(filepath=RAW_DATA_FILEPATH)
    df_clean.to_csv(CLEANED_DATA_FILEPATH)

if __name__ == '__main__':
    run_data_cleaning()