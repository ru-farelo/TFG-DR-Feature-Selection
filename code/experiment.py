import numpy as np
import pandas as pd
from typing import Literal, Union
from sklearn.model_selection import StratifiedKFold
import neptune
from code.model_training import train_a_model
from code.data_processing import load_data
from code.hyperparameter_tuning import grid_search_hyperparams
from code.neptune_utils import upload_preds_to_neptune
import code.metrics as metrics


def run_experiment(
    dataset: str = None,
    classifier: str = None,
    pu_learning: Union[Literal["similarity", "threshold"], bool] = False,
    fast_mrmr: bool = False,
    fast_mrmr_percentage: float = 0.0,
    search_space: dict = None,
    random_state: int = 42,
    neptune_run: Union[neptune.Run, None] = None,
    bagging: bool = False,
    bagging_n: float = 0.0,
    bagging_groups: int = 5
):
    random_state = int(random_state)

    x, y, gene_names = load_data(dataset)

    fast_mrmr_k = fast_mrmr_percentage

    bagging_percentage = bagging_n if bagging else 0.0

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    experiment_preds, experiment_metrics = [], []

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        print(f"\n===== Fold {k} =====")
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"🔍 x_train.shape antes de selección: {x_train.shape}")
        print(f"🔍 y_train distribución: {np.bincount(y_train)}")

        best_params = grid_search_hyperparams(
            x_train,
            y_train,
            classifier=classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            fast_mrmr=fast_mrmr,
            fast_mrmr_k=fast_mrmr_k,
            bagging=bagging,
            bagging_n=bagging_percentage,
            bagging_groups=bagging_groups,
            search_space=search_space,
            neptune_run=neptune_run,
        )

        pred_test = train_a_model(
            x_train,
            y_train,
            x_test,
            classifier=classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_k=best_params.get("pu_k"),
            pul_t=best_params.get("pu_t"),
            fast_mrmr=fast_mrmr,
            fast_mrmr_k=fast_mrmr_k,
            bagging=bagging,
            bagging_n=bagging_percentage,
            bagging_groups=bagging_groups
        )

        experiment_preds += zip(test_idx, gene_names[test_idx], pred_test)
        fold_metrics = metrics.log_metrics(
            y_test, pred_test, neptune_run=neptune_run, run_number=random_state, fold=k
        )
        experiment_metrics.append(fold_metrics)

    experiment_preds = pd.DataFrame(experiment_preds, columns=["id", "gene", "prob"])

    if neptune_run:
        upload_preds_to_neptune(
            preds=experiment_preds,
            random_state=random_state,
            neptune_run=neptune_run,
        )

    for metric in experiment_metrics[0].keys():
        avg_val = np.mean([fold_metrics[metric] for fold_metrics in experiment_metrics])
        if neptune_run:
            neptune_run[f"metrics/run_{random_state}/avg/test/{metric}"] = avg_val
        else:
            print(f"metrics/run_{random_state}/avg/test/{metric}: {avg_val}")

    return experiment_metrics, experiment_preds
