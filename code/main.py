import os
import neptune
import numpy as np
import pandas as pd
from code.config import read_config
from code.experiment import run_experiment

if __name__ == "__main__":

    SEEDS = [14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903]

    args = read_config()

    if args["neptune"]:
        neptune_run = neptune.init_run(
            project="",
            api_token="",
        )
        neptune_run["parameters"] = neptune.utils.stringify_unsupported(args)

    run_metrics_list = []
    run_preds_list = []

    search_space = {
        "pu_k": args["pu_k"],
        "pu_t": args["pu_t"],
    }

    for seed in SEEDS:
        run_metrics, run_preds = run_experiment(
            dataset=args["dataset"],
            classifier=args["classifier"],
            pu_learning=args["pu_learning"],
            fast_mrmr=args["fast_mrmr"],
            fast_mrmr_percentage=args["fast_mrmr_k"],
            search_space=search_space,
            random_state=seed,
            neptune_run=neptune_run if args["neptune"] else None,
            bagging=args["bagging"],
            bagging_n=args["bagging_n"],
            bagging_groups=args["bagging_groups"]
        )
        run_metrics_list.append(run_metrics)
        run_preds_list.append(run_preds)

    for metric in run_metrics_list[0][0].keys():
        avg = np.mean([np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list])
        if args["neptune"]:
            neptune_run[f"metrics/avg/test/{metric}"] = avg
        else:
            print(f"metrics/avg/test/{metric}: {avg}")

    # Average predictions
    for i in range(1, len(run_preds_list)):
        run_preds_list[i] = run_preds_list[i].sort_values(by="gene")

    avg_preds = run_preds_list[0].copy()
    for run_preds in run_preds_list[1:]:
        avg_preds["prob"] += run_preds["prob"]
    avg_preds["prob"] /= len(run_preds_list)
    avg_preds = avg_preds.sort_values(by="prob", ascending=False)
    avg_preds = avg_preds.drop(columns=["id"], errors="ignore")

    if args["neptune"]:
        avg_preds.to_csv("avg_probs.csv", index=False)
        neptune_run["predictions/avg"].upload("avg_probs.csv")
        neptune_run.stop()

    # Cleanup temporary files
    for f in ["preds_temp.csv", "avg_probs.csv"]:
        if os.path.exists(f):
            os.remove(f)
