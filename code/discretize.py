import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# Cargar tu archivo
df = pd.read_csv("../data/CoexpressionDataset.csv")

# Separar las columnas
col_id = df.columns[0]       # Primera columna (ID)
col_label = df.columns[-1]   # Última columna (label)

# Features a discretizar (todas menos la primera y la última)
features = df.columns[1:-1]

# Discretizador: 10 bins, 'ordinal' para mantenerlos como ints, 'uniform' o 'quantile' según prefieras
kbin = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_binned = kbin.fit_transform(df[features])

# Crear el nuevo dataframe discretizado
df_binned = df.copy()
df_binned.loc[:, features] = X_binned.astype(int)  # Solo sobreescribe las features

# Guardar el nuevo archivo
df_binned.to_csv("../data/CoexpressionDataset_binned_normal2.csv", index=False)

print("¡Listo! Archivo discretizado guardado como '.")
