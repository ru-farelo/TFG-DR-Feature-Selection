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
    
