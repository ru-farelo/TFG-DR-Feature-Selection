from typing import Literal, Union
import numpy as np
import pandas as pd
from code.models import get_model, set_class_weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

distances = None

def compute_pairwise_jaccard_measures(x):
    global distances
    x = x.to_numpy().astype(bool)
    distances = pairwise_distances(x, metric="jaccard")
    distances[np.diag_indices_from(distances)] = np.nan


def select_reliable_negatives(
    x: pd.DataFrame,
    y: np.ndarray,
    method: Literal["similarity", "threshold"],
    k: int,
    t: float,
    random_state: int,
) -> tuple:
    global distances

    k, t = int(k), float(t)
    assert 0 <= t <= 1, "t must be between 0 and 1"
    assert k > 0, "k must be greater than 0"

    # Fix index misalignment
    x = x.reset_index(drop=True)
    if isinstance(y, (np.ndarray, list)):
        y = pd.Series(y)
    y = y.reset_index(drop=True)

    p = x[y == 1]
    u = x[y == 0]

    if method == "similarity":
        distances_subset = distances.copy()
        mask = np.isin(np.arange(distances_subset.shape[1]), x.index)
        distances_subset[:, ~mask] = np.nan

        topk = np.argsort(distances_subset, axis=1)[:, :k]
        topk_is_unlabelled = np.isin(topk, u.index)
        closest_unlabelled = topk_is_unlabelled[:, 0]
        topk_percent_unlabelled = np.mean(topk_is_unlabelled, axis=1) >= t

        rn_indices = np.where(closest_unlabelled & topk_percent_unlabelled)[0]
        rn_indices = np.intersect1d(rn_indices, u.index)
        rn = x.loc[rn_indices]

    elif method == "threshold":
        u = u.sample(frac=1.0, random_state=random_state)
        u_split = np.array_split(u, k)

        rn = []

        for subset in u_split:
            x_i = pd.concat([p, subset])
            y_i = np.concatenate([np.ones(len(p)), np.zeros(len(subset))])

            model = RandomForestClassifier(random_state=random_state, min_samples_leaf=5)
            model.fit(x_i, y_i)

            probs = model.predict_proba(subset)[:, 1]
            rn.append(subset[probs <= t])

        rn = pd.concat(rn, axis=0)

    x_new = pd.concat([p, rn], axis=0)
    y_new = np.concatenate([np.ones(len(p)), np.zeros(len(rn))])

    idx = np.arange(len(y_new))
    np.random.shuffle(idx)

    x_final = x_new.iloc[idx].reset_index(drop=True)
    y_final = y_new[idx]

    return x_final, y_final
