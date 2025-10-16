from typing import Literal, Union
import numpy as np
import pandas as pd
import hashlib
from code.models import get_model, set_class_weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

# Cache global para distancias de Jaccard
_distance_cache = {}
distances = None

def _compute_dataset_hash(x):
    """Compute a hash of the dataset to use as cache key."""
    x_array = x.to_numpy().astype(bool)
    # Use shape and a sample of values to create a lightweight hash
    shape_str = f"{x_array.shape[0]}x{x_array.shape[1]}"
    # Sample some values to ensure we're caching the right dataset
    sample_indices = np.linspace(0, x_array.size - 1, min(1000, x_array.size), dtype=int)
    sample_values = x_array.flat[sample_indices].tobytes()
    hash_input = shape_str.encode() + sample_values
    return hashlib.md5(hash_input).hexdigest()

def compute_pairwise_jaccard_measures(x):
    """
    Compute pairwise Jaccard distances with intelligent caching.
    Cache is based on dataset content hash to avoid redundant calculations.
    """
    global distances, _distance_cache
    
    # Compute hash of the dataset
    dataset_hash = _compute_dataset_hash(x)
    
    # Check if we already computed distances for this dataset
    if dataset_hash in _distance_cache:
        distances = _distance_cache[dataset_hash]
        print(f"  Using cached Jaccard distances (hash: {dataset_hash[:8]}...)")
        return
    
    # Compute distances (expensive operation)
    print(f"  Computing Jaccard distances for {x.shape[0]} samples × {x.shape[1]} features...")
    x_array = x.to_numpy().astype(bool)
    distances = pairwise_distances(x_array, metric="jaccard")
    distances[np.diag_indices_from(distances)] = np.nan
    
    # Store in cache
    _distance_cache[dataset_hash] = distances
    print(f"  Distances cached (hash: {dataset_hash[:8]}...)")


def clear_distance_cache():
    """Clear the distance cache. Useful between different experiments/datasets."""
    global _distance_cache
    _distance_cache.clear()


def get_cache_info():
    """Get information about the current cache state."""
    global _distance_cache
    return {
        "cached_datasets": len(_distance_cache),
        "cache_keys": list(_distance_cache.keys())
    }


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