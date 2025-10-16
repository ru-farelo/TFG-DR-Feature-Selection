import numpy as np
import pandas as pd
from typing import Literal, Union
from sklearn.model_selection import StratifiedKFold
import neptune
from code.model_training import train_a_model
from code.data_processing import load_data
from code.hyperparameter_tuning import grid_search_hyperparams
from code.neptune_utils import upload_preds_to_neptune, upload_importances_to_neptune
from code.neptune_utils import upload_emissions_to_neptune
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
    bagging_groups: int = 5,
    extract_importances: bool = False,  # NEW PARAMETER
):
    random_state = int(random_state)

    x, y, gene_names = load_data(dataset)

    fast_mrmr_k = fast_mrmr_percentage

    bagging_percentage = bagging_n if bagging else 0.0

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    experiment_preds, experiment_metrics, experiment_importances = [], [], []
    experiment_emissions_list = []

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        print(f"\n===== Fold {k} =====")
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"x_train.shape antes de selección: {x_train.shape}")
        print(f"y_train distribución: {np.bincount(y_train)}")

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

        result = train_a_model(
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
            bagging_groups=bagging_groups,
            return_importances=extract_importances,
        )
        
        if extract_importances:
            pred_test, fold_importances, fold_emissions = result
            if fold_importances is not None:
                fold_importances['fold'] = k
                experiment_importances.append(fold_importances)
        else:
            pred_test, fold_emissions = result

        # Collect emissions for aggregation (no subir por fold, solo al final)
        if isinstance(fold_emissions, dict):
            experiment_emissions_list.append(fold_emissions)

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

    # Procesar importancias de características si se extrajeron (índice de Gini)
    # Aplicar el mismo patrón correcto: concat + groupby + media adecuada
    avg_importances = None
    if experiment_importances:
        all_importances = pd.concat(experiment_importances, ignore_index=True)
        avg_importances = all_importances.groupby('feature')['gini_importance'].mean().reset_index()
        
        # Normalizar por máximo (no por suma) para evitar valores > 1.0
        max_importance = avg_importances['gini_importance'].max()
        if max_importance > 0:
            # Normalizar a [0, 1] dividiendo por el máximo
            avg_importances['gini_importance'] = avg_importances['gini_importance'] / max_importance
        
        avg_importances = avg_importances.sort_values('gini_importance', ascending=False)
        
        # Upload feature importances to Neptune
        if neptune_run:
            upload_importances_to_neptune(
                importances=avg_importances,
                random_state=random_state,
                neptune_run=neptune_run,
            )
        
        print(f"Gini-based feature importances extracted for seed {random_state} ({len(avg_importances)} features)")
        print(f"   Importance range: [{avg_importances['gini_importance'].min():.4f}, {avg_importances['gini_importance'].max():.4f}]")

    for metric in experiment_metrics[0].keys():
        avg_val = np.mean([fold_metrics[metric] for fold_metrics in experiment_metrics])
        if neptune_run:
            neptune_run[f"metrics/run_{random_state}/avg/test/{metric}"] = avg_val
        else:
            print(f"metrics/run_{random_state}/avg/test/{metric}: {avg_val}")

    # Aggregate emissions across folds if available
    emissions_summary = None
    if experiment_emissions_list:
        # phases union
        phases = set().union(*[set(d.keys()) for d in experiment_emissions_list])
        per_phase_vals = {p: [] for p in phases}
        for d in experiment_emissions_list:
            for p in phases:
                per_phase_vals[p].append(float(d.get(p, 0.0)))

        emissions_summary = {
            p: {"avg_g": float(np.mean(vals)), "total_g": float(np.sum(vals))}
            for p, vals in per_phase_vals.items()
        }

        # Upload SOLO el agregado por run (no por fold)
        if neptune_run:
            for p, stats in emissions_summary.items():
                neptune_run[f"emissions/run_{random_state}/{p}/avg_g"] = stats["avg_g"]
                neptune_run[f"emissions/run_{random_state}/{p}/total_g"] = stats["total_g"]

    return experiment_metrics, experiment_preds, avg_importances, emissions_summary
    