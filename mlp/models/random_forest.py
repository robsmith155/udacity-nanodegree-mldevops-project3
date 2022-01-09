from imblearn.ensemble import BalancedRandomForestClassifier


def train_model(X_train, y_train, config):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    config : dict
        Dictionary containing the model hyperparameters
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = BalancedRandomForestClassifier(
        n_estimators=config.models.random_forest.n_estimators,
        min_samples_split=config.models.random_forest.min_samples_split,
        min_samples_leaf=config.models.random_forest.min_samples_leaf,
        max_features=config.models.random_forest.max_features,
        max_depth=config.models.random_forest.max_depth,
    )
    model.fit(X_train, y_train)
    return model
