import itertools
from typing import Literal, Union
from code.model_training import cv_train_with_params


def get_hyperparam_combinations(search_space):
    """
    Get all possible combinations of hyperparameters to be tested.
    """
    temp = search_space.copy()

    for key, value in temp.items():
        if not isinstance(value, list):
            temp[key] = [value]

    return [dict(zip(temp, values)) for values in itertools.product(*temp.values())]


def validate_PUL_thresholds(method: str = None, k: int = None, t: float = None):
    if method == "similarity":
        if (
            (k == 3 and t not in [0.666, 1])
            or (k == 5 and t not in [4 / 5, 1])
            or (k == 8 and t not in [3 / 4, 7 / 8, 1])
        ):
            return False

    elif method == "threshold":
        if (k == 1 and t not in [0.05, 0.1, 0.15, 0.2, 0.25]) or (
            k == 3 and t not in [0.1, 0.2, 0.3, 0.4]
        ):
            return False

    return True


def grid_search_hyperparams(
    x_train,
    y_train,
    classifier: Union[Literal["EEC", "BRF", "CAT", "XGB"], None] = None,
    random_state=42,
    pu_learning=False,
    fast_mrmr=False,
    fast_mrmr_k=None, 
    bagging=False,         
    bagging_n=None,  
    bagging_groups=5,         
    search_space: dict = None,
    neptune_run=None,
):
    """Perform grid search."""

    best_config = {"params": None, "score": 0}
    hyperparam_combinations = get_hyperparam_combinations(search_space)

    if len(hyperparam_combinations) == 1:
        return {k: v[0] if isinstance(v, list) else v for k, v in search_space.items()}

    for params in hyperparam_combinations:
        if pu_learning and not validate_PUL_thresholds(
            pu_learning,
            params.get("pu_k"),
            params.get("pu_t"),
        ):
            continue

        score = cv_train_with_params(
            x_train,
            y_train,
            classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            fast_mrmr=fast_mrmr,
            fast_mrmr_k=fast_mrmr_k,  
            pul_k=params.get("pu_k"),
            pul_t=params.get("pu_t"),
            bagging=bagging,           
            bagging_n=bagging_n ,  
            bagging_groups=bagging_groups     
        )

        if score > best_config["score"]:
            best_config = {"params": params, "score": score}

    # Log or print best parameters
    for key, value in best_config["params"].items():
        if neptune_run:
            neptune_run[f"parameters/best_selected/{key}"] = value
        else:
            print(f"parameters/{key}: {value}")

    return best_config["params"]
