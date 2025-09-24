import time
from typing import Literal, Union
import neptune
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from code.models import get_model, set_class_weights
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
import code.pu_learning as pul
from code.execute_fastmrmr import execute_fast_mrmr_pipeline
from code.bagging_disjoint import bagging_feature_selection_disjoint

CV_INNER = 5

def cv_train_with_params(
    x_train,
    y_train,
    classifier,
    random_state,
    pu_learning=False,
    pul_k=None,
    pul_t=None,
    fast_mrmr=False,
    fast_mrmr_k=None,
    bagging=False,
    bagging_n=None,
    bagging_groups=5, 
):

    inner_skf = StratifiedKFold(n_splits=CV_INNER, shuffle=True, random_state=random_state)
    score = []

    if isinstance(x_train, np.ndarray):
        x_train = pd.DataFrame(x_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    for _, (learn_idx, val_idx) in enumerate(inner_skf.split(x_train, y_train)):
        x_learn, x_val = x_train.iloc[learn_idx], x_train.iloc[val_idx]
        y_learn, y_val = y_train.iloc[learn_idx], y_train.iloc[val_idx]

        pred_val = train_a_model(
            x_learn,
            y_learn,
            x_val,
            classifier,
            random_state=random_state,
            pu_learning=pu_learning,
            pul_k=pul_k,
            pul_t=pul_t,
            fast_mrmr=fast_mrmr,
            fast_mrmr_k=fast_mrmr_k,
            bagging=bagging,
            bagging_n=bagging_n,
            bagging_groups=bagging_groups,
        )

        if pu_learning:
            score.append(f1_score(y_val, pred_val > 0.5))
        else:
            score.append(geometric_mean_score(y_val, pred_val > 0.5))

    return np.mean(score)


def train_a_model(
    x_train,
    y_train,
    x_test,
    classifier: Literal["CAT", "BRF", "XGB", "EEC"],
    random_state: int,
    pu_learning: Union[str, bool] = False,
    pul_k: int = None,
    pul_t: float = None,
    fast_mrmr: bool = False,
    fast_mrmr_k: int = 0,
    bagging: bool = False,
    bagging_n: float = 0.0,
    bagging_groups: int = 5,
    max_fastmrmr_retries: int = 8,  
):
    if isinstance(x_train, np.ndarray):
        x_train = pd.DataFrame(x_train)
    if isinstance(x_test, np.ndarray):
        x_test = pd.DataFrame(x_test)
    if isinstance(y_train, (np.ndarray, list)):
        y_train = pd.Series(y_train)

    x_train_proc = x_train.copy()
    x_test_proc = x_test.copy()

    # Bagging disjunto
    if bagging:
        print(f" Bagging disjunto ACTIVADO (n_grupos={bagging_groups}, porcentaje={bagging_n}%)")
        x_train_proc, x_test_proc = bagging_feature_selection_disjoint(
            x_train_proc, x_test_proc, n_groups=bagging_groups, percentage=bagging_n, seed=random_state
        )
        print(f"   → Características tras BAGGING: {x_train_proc.shape[1]}")
        print(f"   → Nombres de las primeras 10 columnas: {list(x_train_proc.columns[:10])}")

    # Fast MRMR con reintentos automáticos
    if fast_mrmr:
        if isinstance(fast_mrmr_k, float) and 0 < fast_mrmr_k < 100:
            n_features = int(x_train_proc.shape[1] * (fast_mrmr_k / 100))
            print(f" Fast-MRMR ACTIVADO (k={n_features}, {fast_mrmr_k}% de {x_train_proc.shape[1]} columnas tras bagging)")
            fast_mrmr_k = n_features
        else:
            print(f" Fast-MRMR ACTIVADO (k={fast_mrmr_k} columnas tras bagging)")
            fast_mrmr_k = int(fast_mrmr_k)

        # Reintenta Fast-MRMR si no selecciona ninguna feature
        for retry in range(max_fastmrmr_retries):
            seed_try = random_state + retry
            x_train_proc_tmp, x_test_proc_tmp = execute_fast_mrmr_pipeline(
                x_train_proc,
                y_train,
                x_test_proc,
                classifier,
                seed_try,
                fast_mrmr_k
            )
            if x_train_proc_tmp.shape[1] > 0:
                if retry > 0:
                    print(f" [REINTENTO]: Fast-MRMR seleccionó features tras {retry+1} intento(s) (seed={seed_try})")
                x_train_proc, x_test_proc = x_train_proc_tmp, x_test_proc_tmp
                break
            else:
                print(f" [REINTENTO]: Fast-MRMR no seleccionó features (seed={seed_try}). Reintentando...")
        else:
            # Si tras todos los reintentos sigue sin seleccionar nada
            print(" [AVISO]: Fast-MRMR no ha seleccionado ninguna feature tras varios intentos. No se puede entrenar el modelo en este fold.")
            print(" [DEBUG] Tamaño de x_train_proc:", x_train_proc_tmp.shape)
            print(" [DEBUG] Saliendo de train_a_model con array vacío.\n")
            return np.zeros(len(x_test_proc))  # <--- Para no romper el pipeline

        print(f"   → Características tras FAST-MRMR: {x_train_proc.shape[1]}")
        print(f"   → Nombres de las primeras 10 columnas: {list(x_train_proc.columns[:10])}")

    print(f" Número de características FINAL antes de entrenamiento: {x_train_proc.shape[1]}")
    print("==========================================================\n")

    if pu_learning:
        pul.compute_pairwise_jaccard_measures(x_train_proc)
        x_train_final, y_train_final = pul.select_reliable_negatives(
            x_train_proc,
            y_train,
            pu_learning,
            pul_k,
            pul_t,
            random_state=random_state,
        )
    else:
        x_train_final = x_train_proc.copy()
        y_train_final = y_train.copy()

    x_test_final = x_test_proc.copy()

    model = get_model(classifier, random_state=random_state)

    if classifier == "CAT":
        model = set_class_weights(model, y_train_final)
        model.fit(x_train_final, y_train_final, verbose=0)
    elif classifier == "XGB":
        pos_weight = (len(y_train_final) - np.sum(y_train_final)) / np.sum(y_train_final)
        model.set_params(scale_pos_weight=pos_weight)
        model.fit(x_train_final, y_train_final, verbose=0)
    else:
        model.fit(x_train_final, y_train_final)

    probs = model.predict_proba(x_test_final)[:, 1]
    return probs
