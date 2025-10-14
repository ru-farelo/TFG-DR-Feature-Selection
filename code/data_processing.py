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

    raw_df = pd.read_csv(f"./data/{dataset_name}.csv")

    x, y = raw_df.iloc[:, 1:-1], raw_df.iloc[:, -1]
    y = np.logical_not(preprocessing.LabelEncoder().fit(y).transform(y)).astype(int)
    gene_names = raw_df.iloc[:, 0]
    return x, y, gene_names
