import time
import os
import subprocess
import numpy as np
import pandas as pd

CV_INNER = 5

def execute_fast_mrmr_pipeline(x_train, y_train, x_test, classifier, random_state, fast_mrmr_k):
    """
    Ejecuta toda la pipeline de selección de características con Fast-MRMR
    """
    csv_path = "./utils/data-reader/data_xtrain_temporaly/x_train.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    y_train_mrmr = y_train.replace({0: -1, 1: 1}).reset_index(drop=True)

    x_train_mrmr = pd.concat([y_train_mrmr, x_train.reset_index(drop=True)], axis=1)
    x_train_mrmr.to_csv(csv_path, index=False, header=False)

    mrmr_path = "./utils/data-reader/data_mrmr/x_train.mrmr"
    os.makedirs(os.path.dirname(mrmr_path), exist_ok=True)
    if os.path.exists(mrmr_path):
        os.remove(mrmr_path)

    #print(" Ejecutando binario Fast-MRMR...")
    subprocess.run("make clean", shell=True, capture_output=True, text=True, cwd="utils/data-reader")
    subprocess.run("make all", shell=True, capture_output=True, text=True, cwd="utils/data-reader")
    time.sleep(1)
    subprocess.run("./mrmr-reader ./data_xtrain_temporaly/x_train.csv", shell=True, capture_output=True, text=True, cwd="utils/data-reader")
    time.sleep(1)
    subprocess.run("make clean", shell=True, capture_output=True, text=True, cwd="src_c")
    subprocess.run("make all", shell=True, capture_output=True, text=True, cwd="src_c")
    #print(" Fast-MRMR ejecutado")
    result = subprocess.run(f"./fast-mrmr -a {fast_mrmr_k}", shell=True, capture_output=True, text=True, cwd="src_c")
    #print("Fast-MRMR output:", result.stdout)

    selected_features_indices = [int(i) - 1 for i in result.stdout.strip().rstrip(',').split(',') if i.strip().isdigit()]
    selected_features = x_train.columns[selected_features_indices]
    #print("selected_features", selected_features)
    #print("Número de columnas en x_train después de Fast-MRMR:", len(selected_features))

    x_train = x_train.loc[:, selected_features]
    x_test = x_test.loc[:, selected_features]

    return x_train, x_test

