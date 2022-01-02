# Script to train machine learning model.
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.processing.data import process_data

# Project Directories
PACKAGE_ROOT = Path(mlp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILEPATH = ROOT / "config.yaml"

# Load config file
with open(CONFIG_FILEPATH, "r", encoding='utf-8') as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

CLEANED_DATA_FILEPATH = ROOT / config.data.cleaned.filepath

# Load data
data = pd.read_csv(CLEANED_DATA_FILEPATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    X=train, categorical_features=config.data_processing.cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    X=test, categorical_features=config.data_processing.cat_features, label="salary", training=False
)

# Train and save a model.
