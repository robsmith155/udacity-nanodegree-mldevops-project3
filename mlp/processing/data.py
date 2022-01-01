import pandas as pd

def clean_data(filepath: str) -> pd.DataFrame:
    """ Initial cleaning of the raw census data.

    This does some basic cleaning of the census data to remove rows with
    missing values and to remove spaces before text.

    Inputs
    ------
    filepath : str
        Filepath of the raw census dataset
    
    Returns
    -------
    df : pd.DataFrame
        Cleaned data.
    """
    df = pd.read_csv(filepath, na_values='?', skipinitialspace=True)
    df.dropna(inplace=True)
    return df