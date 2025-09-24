import numpy as np
import pandas as pd


def bagging_feature_selection_disjoint(x_train, x_test, n_groups=5, percentage=5.0, seed=42):
    """
    Divide features into disjoint groups and select percentage from each group.
    
    Returns: (selected_x_train, selected_x_test)
    """
    np.random.seed(seed)
    total_features = x_train.shape[1]
    cols = np.array(x_train.columns)
    np.random.shuffle(cols)  # Shuffle to randomize the groupings

    group_size = total_features // n_groups
    selected_cols = []

    for i in range(n_groups):
        if i < n_groups - 1:
            group_cols = cols[i * group_size : (i + 1) * group_size]
        else:
            group_cols = cols[i * group_size :]
        num_to_select = max(1, int(len(group_cols) * (percentage / 100)))
        selected = np.random.choice(group_cols, size=num_to_select, replace=False)
        selected_cols.extend(selected)

    print(f"Bagging (disjoint): {len(selected_cols)} columnas seleccionadas de {total_features} ({n_groups} grupos, {percentage}% de cada grupo)")
    return x_train[selected_cols], x_test[selected_cols]
