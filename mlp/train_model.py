# Script to train machine learning model.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from box import Box
from sklearn.model_selection import train_test_split

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.models.random_forest import train_model
from mlp.models.utils import save_model
from mlp.processing.data import process_data

# Project Directories
PACKAGE_ROOT = Path(mlp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILEPATH = ROOT / "config.yaml"

# Load config file
with open(CONFIG_FILEPATH, "r", encoding="utf-8") as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

CLEANED_DATA_FILEPATH = ROOT / config.data.cleaned.filepath


def run_train_model() -> None:
    # Load data
    data = pd.read_csv(CLEANED_DATA_FILEPATH).drop(["Unnamed: 0"], axis=1)
    train, test = train_test_split(
        data, test_size=0.20, random_state=config.models.random_seed
    )

    # Process the training and test data
    X_train, y_train, encoder, lb = process_data(
        X=train,
        categorical_features=config.data_processing.cat_features,
        label=config.data_processing.label,
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        X=test,
        categorical_features=config.data_processing.cat_features,
        label=config.data_processing.label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model.
    model = train_model(X_train, y_train, config)

    # Save model and encoders
    save_model(
        model=encoder, output_filepath=config.data_processing.encoder_filepath
    )
    save_model(
        model=lb, output_filepath=config.data_processing.binarizer_filepath
    )
    save_model(
        model=model,
        output_filepath=config.models.random_forest.output_filepath,
    )

    # Save processed data
    with open(config.data_processing.train_filepath, "wb") as f:
        np.save(f, X_train)
        np.save(f, y_train)

    with open(config.data_processing.test_filepath, "wb") as f:
        np.save(f, X_test)
        np.save(f, y_test)


if __name__ == "__main__":
    run_train_model()
