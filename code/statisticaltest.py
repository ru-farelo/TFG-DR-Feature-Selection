import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

MODEL_1_ID = '133'
MODEL_2_ID = '136'

METRICS = ['sensitivity', 'specificity', 'precision', 'f1', 'gmean','auc_roc']   

data = pd.read_csv('DR-Gene-Prediction.csv')

print(data.columns)

model_1_data = data[data['Id'] == f"DRGEN-{MODEL_1_ID}"]
model_2_data = data[data['Id'] == f"DRGEN-{MODEL_2_ID}"]

for metric in METRICS:
    metric_cols = [col for col in data.columns if metric in col]

    

    model_1_metric_data = model_1_data[metric_cols].values[0]
    model_2_metric_data = model_2_data[metric_cols].values[0]

    print(np.mean(model_1_metric_data), np.std(model_1_metric_data))
    print(np.mean(model_2_metric_data), np.std(model_2_metric_data))

    # print(model_1_metric_data)
    # print(model_2_metric_data)

    t_statistic, p_value = ttest_rel(model_1_metric_data, model_2_metric_data, alternative='greater')

    print(f"Metric: {metric}")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")


                    

