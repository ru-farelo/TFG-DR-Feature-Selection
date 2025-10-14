import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score

data_features = None


def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a CSV file.

    Parameters:
    dataset_path (str): The name of the dataset.

    Returns:
    tuple: A tuple containing the features (x) and the labels (y).
    """

    raw_df = pd.read_csv(f"./Data/{dataset_name}.csv")
    x, y = raw_df.iloc[:, 1:-1], raw_df.iloc[:, -1]
    y = np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int)
    gene_names = raw_df.iloc[:, 0]
    return x, y, gene_names


def store_data_features(x: pd.DataFrame) -> np.ndarray:
    """
    Store the features of the data for later use.

    Parameters:
    x (pd.DataFrame): The features of the data.
    
    Returns:
    np.ndarray: Array of indices with shape (n_samples, 1) to maintain 2D structure
    """
    global data_features
    data_features = x

    # Return indices as 2D array to maintain compatibility with PU learning
    indices = np.arange(len(x)).reshape(-1, 1)
    return indices


def get_data_features(indices) -> pd.DataFrame:
    """
    Get the examples of the data with the specified indices.

    Parameters:
    indices (list, np.ndarray): The indices of the examples to retrieve.
                               Can be 1D or 2D array, will be flattened.

    Returns:
    pd.DataFrame: The examples with the specified indices.
    """
    # Handle different input formats - flatten if necessary
    if isinstance(indices, np.ndarray):
        if indices.ndim > 1:
            indices = indices.flatten()
    elif isinstance(indices, list):
        indices = np.array(indices).flatten()
    
    return data_features.iloc[indices]


def generate_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    binary_threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate features for training and testing data based on specified parameters.

    Parameters:
        - x_train (np.ndarray): Training data indices.
        - x_test (np.ndarray): Testing data indices.
        - binary_threshold (float): Threshold for feature selection.
    Returns:
        - x_train_temp (pd.DataFrame): Transformed training data features.
        - x_test_temp (pd.DataFrame): Transformed testing data features.
    """

    # Get actual features using the indices
    x_train_features = get_data_features(x_train)
    x_test_features = get_data_features(x_test)

    # Apply feature filtering
    x_train_filtered = x_train_features.loc[:, x_train_features.mean() >= binary_threshold]
    x_test_filtered = x_test_features.loc[:, x_train_filtered.columns]    

    return x_train_filtered, x_test_filtered
