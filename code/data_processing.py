import numpy as np
import pandas as pd
from sklearn import preprocessing

data_features = None


def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a CSV file.

    Parameters:
    dataset_path (str): The name of the dataset.

    Returns:
    tuple: A tuple containing the features (x) and the labels (y).
    """
    global data_features

    raw_df = pd.read_csv(f"./data/{dataset_name}.csv")

    x, y = raw_df.iloc[:, 1:-1], raw_df.iloc[:, -1]
    # Keep y as pandas Series instead of converting to numpy array
    y = pd.Series(np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int), 
                  index=y.index, name=y.name)
    gene_names = raw_df.iloc[:, 0]
    
    # Store full features data globally for PU learning access
    data_features = x.copy()
    
    return x, y, gene_names


def get_data_features(indices):
    """
    Get features for specific gene indices from the global data_features.
    
    Parameters:
    indices (array-like): Indices of genes to retrieve features for
    
    Returns:
    pd.DataFrame: Features for the specified gene indices
    """
    global data_features
    if data_features is None:
        raise ValueError("No data loaded. Call load_data first.")
    
    # Convert indices to list if it's a numpy array
    if hasattr(indices, 'tolist'):
        indices = indices.tolist()
    
    # Use iloc to get features for the specified indices
    return data_features.iloc[indices]

