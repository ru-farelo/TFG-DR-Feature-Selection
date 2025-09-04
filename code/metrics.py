import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from typing import Union
import neptune


def recall_at_k(y_true, y_prob, k: int = 10):
    """
    Calculate the recall at k

    Parameters
    ----------
    - y_true : array-like
      - True labels.
    - y_prob : array-like
      - Predicted probabilities.
    - k : int
      - The number of top predictions to consider.

    Returns:
    - recall : float
      - The recall at k.

    """

    sort = y_prob.argsort()[::-1][:k]
    recall_at_k = np.sum(y_true[sort]) / np.sum(y_true)

    return recall_at_k


def precision_at_k(y_true, y_prob, k: int = 10):
    """
    Calculate the precision at k

    Parameters
    ----------
    - y_true : array-like
      - True labels.
    - y_prob : array-like
      - Predicted probabilities.
    - k : int
      - The number of top predictions to consider.

    Returns:
    - precision : float
      - The precision at k.

    """

    sort = y_prob.argsort()[::-1][:k]
    precision_at_k = np.sum(y_true[sort]) / k

    return precision_at_k


def log_metrics(
    y_true,
    y_prob,
    neptune_run: Union[None, neptune.Run] = None,
    run_number: int = None,
    fold: int = Union[int, None],
) -> dict:
    """
    Calculate and log various evaluation metrics based on the true labels and predicted probabilities.

    Parameters
    ----------
    - y_true : array-like
      - True labels.
    - y_prob : array-like
      - Predicted probabilities.
    - neptune_run : neptune.Run
      - Neptune run object for logging metrics
    - run_number : int
      - Run identifier (usually the random state)
    - fold : int | None
      - Fold identifier, if it's a single fold's metrics

    Returns:
    - metrics (dict): Dictionary containing the calculated evaluation metrics.

    """
    y_pred = (y_prob >= 0.5).astype(
        int
    )  # Threshold probabilities to obtain binary predictions

    # Binary classification metrics
    test_sens = sensitivity_score(y_true, y_pred)
    test_spec = specificity_score(y_true, y_pred)
    test_gmean = geometric_mean_score(y_true, y_pred)
    test_auc_roc = roc_auc_score(y_true, y_prob)
    test_precision = precision_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    test_auc_pr = auc(recalls, precisions)

    # Ranking metrics
    test_ndcg_at_k = ndcg_score(y_true.reshape(1, -1), y_prob.reshape(1, -1), k=10)
    test_recall_at_k = recall_at_k(y_true, y_prob, k=10)
    test_precision_at_k = precision_at_k(y_true, y_prob, k=10)

    metrics = {
        "sensitivity": test_sens,
        "specificity": test_spec,
        "precision": test_precision,
        "f1": test_f1,
        "gmean": test_gmean,
        "auc_roc": test_auc_roc,
        "auc_pr": test_auc_pr,
        "recall_at_10": test_recall_at_k,
        "ndcg_at_10": test_ndcg_at_k,
        "precision_at_10": test_precision_at_k,
    }

    if fold is not None:
        for metric, value in metrics.items():
            if neptune_run:
                neptune_run[f"metrics/run_{run_number}/fold_{fold}/test/{metric}"] = (
                    value
                )
            else:
                print(f"metrics/run_{run_number}/fold_{fold}/test/{metric}: {value}")

    return metrics