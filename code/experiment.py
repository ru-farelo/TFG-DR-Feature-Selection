import numpy as np
import pandas as pd
from typing import Literal, Union
from sklearn.model_selection import StratifiedKFold
import neptune
from codecarbon import EmissionsTracker
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
    bagging_groups: int = 5,
    extract_importances: bool = False,  # NEW PARAMETER
    tracker: EmissionsTracker = None    # NEW: CO2 tracker instance
):
    random_state = int(random_state)

    x, y, gene_names = load_data(dataset)

    # 🔬 PHASE 1: Feature/Instance Selection (CO2 measured separately)
    # - Fast-mRMR = Feature Selection (selects best features)
    # - PU Learning = Instance Selection (selects reliable negative instances)
    
    if tracker and random_state == 14:  # Only measure once per experiment
        if fast_mrmr:
            # Fast-mRMR: Feature Selection
            tracker.start_task("feature_selection")
            print("🔬 Measuring CO2 for Feature Selection (Fast-mRMR)")
        elif pu_learning:
            # PU Learning: Instance Selection  
            tracker.start_task("instance_selection")
            print("🔬 Measuring CO2 for Instance Selection (PU Learning)")
        
        if fast_mrmr or pu_learning:
            if fast_mrmr:
                tracker.stop_task("feature_selection")
            else:
                tracker.stop_task("instance_selection")

    fast_mrmr_k = fast_mrmr_percentage

    bagging_percentage = bagging_n if bagging else 0.0

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    experiment_preds, experiment_metrics, experiment_importances = [], [], []

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        print(f"\n===== Fold {k} =====")
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

        # 🚀 PHASE 2: Training (CO2 measured separately)
        if tracker:
            tracker.start_task("training")
        
        # Use pre-tuned hyperparameters (no inner CV during 10-fold CV measurement)
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
            tracker=tracker  # Pass tracker for training/inference separation
        )
        
        if tracker:
            tracker.stop_task("training")
        
        if extract_importances:
            pred_test, fold_importances = result
            if fold_importances is not None:
                fold_importances['fold'] = k
                experiment_importances.append(fold_importances)
        else:
            pred_test = result

        # 🎯 PHASE 3: Inference/Metrics (CO2 measured separately)
        if tracker:
            tracker.start_task("inference")
        
        experiment_preds += zip(test_idx, gene_names[test_idx], pred_test)
        fold_metrics = metrics.log_metrics(
            y_test, pred_test, neptune_run=neptune_run, run_number=random_state, fold=k
        )
        experiment_metrics.append(fold_metrics)
        
        if tracker:
            tracker.stop_task("inference")

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

    # Process feature importances if extracted (Gini-based importance from ensemble)
    # Apply same improvements as main.py: proper normalization without clipping
    avg_importances = None
    if experiment_importances:
        all_importances = pd.concat(experiment_importances, ignore_index=True)
        avg_importances = all_importances.groupby('feature')['gini_importance'].mean().reset_index()
        
        # Apply reference pattern: normalize by max value instead of clipping
        max_importance = avg_importances['gini_importance'].max()
        if max_importance > 0:
            # Normalize to [0, 1] by dividing by max (no arbitrary clipping)
            avg_importances['gini_importance'] = avg_importances['gini_importance'] / max_importance
        
        avg_importances = avg_importances.sort_values('gini_importance', ascending=False)
        
        # Upload feature importances to Neptune
        if neptune_run:
            importances_sorted = avg_importances.sort_values(by="gini_importance", ascending=False)
            importances_sorted.to_csv("feature_importances_temp.csv", index=False)
            neptune_run[f"feature_importances/run_{random_state}"].upload("feature_importances_temp.csv")
        
        print(f"Gini-based feature importances extracted for seed {random_state} ({len(avg_importances)} features)")
        print(f"   Importance range: [{avg_importances['gini_importance'].min():.4f}, {avg_importances['gini_importance'].max():.4f}]")

    # Stop CO2e tracking after 10-fold CV completion (if enabled)
    cv_emissions = None
    if tracker is not None:
        cv_emissions = tracker.stop()
        print(f"🌱 Nested CV completed: {cv_emissions * 1000:.2f} g CO2e (training + inference only)")

    return experiment_metrics, experiment_preds, avg_importances, cv_emissions
