import pickle


def save_model(model, output_filepath: str) -> None:
    """
    Save the trained model to the specified filepath.

    Inputs
    ------
    model : 
        Trained Scikit Learn model or Imbalanced-learn model.
    output_filepath : str
        File path to save the model.
    Returns
    -------
    None
    """
    pickle.dump(model, open(output_filepath, 'wb'))

