import os
import neptune
import pandas as pd
from typing import Union


def upload_preds_to_neptune(
    preds: pd.DataFrame = None,
    random_state: Union[int, None] = None,
    neptune_run: neptune.Run = None,
):
    """
    Upload to Neptune the DR-Relatedness predictions for the current run.

    Parameters
    ----------
    - preds : pd.DataFrame
        - The predictions DataFrame.
    - random_state : int | None
        - The number of the run (random state used)
        - If None, it is assumed these are the average predictions.
    - neptune_run : neptune.Run
        - The Neptune run object.
    """
    preds_sorted = preds.sort_values(by="prob", ascending=False)
    preds_sorted.to_csv("preds_temp.csv", index=False)
    neptune_run[f"predictions/run_{random_state}"].upload("preds_temp.csv")
    


def upload_importances_to_neptune(
    importances: pd.DataFrame = None,
    random_state: Union[int, None] = None,
    neptune_run: neptune.Run = None,
):
    """
    Upload to Neptune the feature importances for the current run.

    Parameters
    ----------
    - importances : pd.DataFrame
        - The feature importances DataFrame.
    - random_state : int | None
        - The number of the run (random state used)
        - If None, it is assumed these are the average importances.
    - neptune_run : neptune.Run
        - The Neptune run object.
    """
    importances_copy = importances.copy()
    
    # Normalizar importancias por MÁXIMO (no por suma)
    max_importance = importances_copy['gini_importance'].max()
    if max_importance > 0:
        importances_copy['gini_importance'] = importances_copy['gini_importance'] / max_importance
    
    importances_sorted = importances_copy.sort_values(by="gini_importance", ascending=False)
    importances_sorted.to_csv("importances_temp.csv", index=False)
    neptune_run[f"feature_importances/run_{random_state}"].upload("importances_temp.csv")
    

def upload_emissions_to_neptune(
    emissions_per_phase: dict = None,
    random_state: Union[int, None] = None,
    neptune_run: neptune.Run = None,
):
    """Upload emissions (in grams) per pipeline phase to Neptune.

    emissions_per_phase: dict mapping phase_name -> grams (float)
    """
    if emissions_per_phase is None or neptune_run is None:
        return

    for phase, grams in emissions_per_phase.items():
        # Store as a numeric metric under a clear namespace
        key = f"emissions/run_{random_state}/{phase}"
        try:
            neptune_run[key] = float(grams)
        except Exception:
            # Fallback to string representation if direct assignment fails
            neptune_run[key] = str(grams)
