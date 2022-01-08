import sys
from pathlib import Path
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
import numpy as np
from typing import Tuple
import imblearn

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.processing.data import clean_data, process_data
from mlp.models.random_forest import train_model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: imblearn.ensemble._forest.BalancedRandomForestClassifier , X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Scikit-learn or imbalanced-learn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
