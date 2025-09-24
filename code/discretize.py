import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# Load the dataset
df = pd.read_csv("../data/CoexpressionDataset.csv")

# Separate the columns
col_id = df.columns[0]       # First column (ID)
col_label = df.columns[-1]   # Last column (label)

# Features to discretize (all except the first and last)
features = df.columns[1:-1]

# Discretizer: 10 bins, 'ordinal' to keep them as ints, 'uniform' or 'quantile' depending on your preference
kbin = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_binned = kbin.fit_transform(df[features])

# Create the new discretized dataframe
df_binned = df.copy()
df_binned.loc[:, features] = X_binned.astype(int)  # Only overwrite the features

# Save the new file
df_binned.to_csv("../data/CoexpressionDataset_binned_normal2.csv", index=False)

print("¡Listo! Archivo discretizado guardado como '.")
