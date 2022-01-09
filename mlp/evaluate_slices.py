import sys
import pickle
from pathlib import Path
import yaml
from box import Box
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

cwd = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, cwd)

import mlp
from mlp.processing.data import process_data
from mlp.models.evaluate import inference, compute_model_metrics

# Project Directories
PACKAGE_ROOT = Path(mlp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILEPATH = ROOT / "config.yaml"

# Load config file
with open(CONFIG_FILEPATH, "r", encoding='utf-8') as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

CLEANED_DATA_FILEPATH = ROOT / config.data.cleaned.filepath
SLICE_OUTPUT_FILEPATH = ROOT / config.models.metrics.slice_filepath
ALL_TRAINING_OUTPUT_FILEPATH = ROOT / config.models.metrics.all_train_filepath
ALL_TEST_OUTPUT_FILEPATH = ROOT / config.models.metrics.all_test_filepath

def run_evaluate_slice_scores() -> None:
    """
    Slices the test data for each category in each categorical feature and 
    computes the model metric scores. These are then saved to a file in the 
    outputs folder.
    """
    categorical_features = config.data_processing.cat_features
    label = config.data_processing.label
    ohe_encoder_filepath = config.data_processing.encoder_filepath
    binarizer_filepath = config.data_processing.binarizer_filepath
    model_filepath = config.models.random_forest.output_filepath

    clean_df = pd.read_csv(CLEANED_DATA_FILEPATH).drop(['Unnamed: 0'],axis=1)
    ohe_encoder = pickle.load(open(ohe_encoder_filepath, 'rb'))
    label_binarizer = pickle.load(open(binarizer_filepath, 'rb'))
    model = pickle.load(open(model_filepath, 'rb'))

    all_scores_df = pd.DataFrame(columns=['feature', 'category', 'num_samples', 'precision', 'recall', 'fbeta'])
    
    _, test = train_test_split(clean_df, test_size=0.20, random_state=config.models.random_seed)

    for feature in categorical_features:
        for category in test[feature].unique():
            filtered_df = test[test[feature] == category]
            num_samples = len(filtered_df)

            # Process filtered data
            X_test, y_test, _, _ = process_data(
                X=filtered_df, 
                categorical_features=categorical_features, 
                label=label, 
                training=False,
                encoder=ohe_encoder,
                lb=label_binarizer)

            # Make predictions and score
            preds = inference(model=model, X=X_test)
            scores = compute_model_metrics(y=y_test, preds=preds)
            scores_list = [feature, category, num_samples, scores[0], scores[1], scores[2]]
            scores_series = pd.Series(scores_list, index=all_scores_df.columns)

            # Add scores to DataFrame
            all_scores_df = all_scores_df.append(scores_series, ignore_index=True)

    all_scores_df.to_csv(SLICE_OUTPUT_FILEPATH)

def run_evaluate_all() -> None:
    """
    Output the performance metrics for all data for both the training and test
    datasets.
    """
    train_data_filepath = config.data_processing.train_filepath
    test_data_filepath = config.data_processing.test_filepath
    model_filepath = config.models.random_forest.output_filepath

    with open(train_data_filepath, 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)
    
    with open(test_data_filepath, 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)

    model = pickle.load(open(model_filepath, 'rb'))

    # Make predictions and score
    train_preds = inference(model=model, X=X_train)
    test_preds = inference(model=model, X=X_test)
    train_scores = compute_model_metrics(y=y_train, preds=train_preds)
    test_scores = compute_model_metrics(y=y_test, preds=test_preds)

    # Save results
    train_results = {'precision': train_scores[0],
                     'recall': train_scores[1],
                     'fbeta': train_scores[2]}

    test_results = {'precision': test_scores[0],
                    'recall': test_scores[1],
                    'fbeta': test_scores[2]}

    with open(ALL_TRAINING_OUTPUT_FILEPATH, 'w') as f:
        f.write(str(train_results))
    with open(ALL_TEST_OUTPUT_FILEPATH, 'w') as f:
        f.write(str(test_results))

if __name__ == '__main__':
    run_evaluate_slice_scores()
    run_evaluate_all()
