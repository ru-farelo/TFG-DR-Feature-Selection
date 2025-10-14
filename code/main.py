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
    run_importances_list = []  # Store feature importances from each run

    search_space = {
        "pu_k": args["pu_k"],
        "pu_t": args["pu_t"],
    }

    for seed in SEEDS:
        run_metrics, run_preds, run_importances = run_experiment(
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
            bagging_groups=args["bagging_groups"],
            extract_importances=True,  # Enable feature importance extraction
        )
        
        run_metrics_list.append(run_metrics)
        run_preds_list.append(run_preds)
        
        if run_importances is not None:
            run_importances['seed'] = seed
            run_importances_list.append(run_importances)

    for metric in run_metrics_list[0][0].keys():
        avg = np.mean([np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list])
        if args["neptune"]:
            neptune_run[f"metrics/avg/test/{metric}"] = avg
        else:
            print(f"metrics/avg/test/{metric}: {avg}")

    # ✅ CORRECCIÓN: Calcular media correcta por gene ID (no por posición)
    print(f"📊 Calculando predicciones promedio para {len(run_preds_list)} seeds...")
    print(f"📋 Ejemplo DataFrames antes del promedio:")
    for i, df in enumerate(run_preds_list[:2]):  # Show first 2 for debugging
        print(f"   Seed {i+1}: {len(df)} genes, prob range [{df['prob'].min():.4f}, {df['prob'].max():.4f}]")
    
    # MÉTODO CORRECTO: concat + groupby por gene
    all_preds = pd.concat(run_preds_list, ignore_index=True)
    avg_preds = all_preds.groupby('gene')['prob'].mean().reset_index()
    
    # Add id column if needed (using the gene as id)
    avg_preds['id'] = range(len(avg_preds))
    
    # Sort by probability descending
    avg_preds = avg_preds.sort_values(by="prob", ascending=False)
    
    print(f"✅ Predicciones promedio calculadas:")
    print(f"   📈 Total genes: {len(avg_preds)}")
    print(f"   📊 Prob range: [{avg_preds['prob'].min():.4f}, {avg_preds['prob'].max():.4f}]")
    print(f"   🎯 Genes con prob > 0.5: {len(avg_preds[avg_preds['prob'] > 0.5])}")
    print(f"   🔝 Top 10 probabilidades: {avg_preds['prob'].head(10).tolist()}")
    
    avg_preds = avg_preds.drop(columns=["id"], errors="ignore")

    # ✅ CORRECCIÓN: Calcular importancias promedio por feature ID (índice de Gini)
    # Aplicar el mismo patrón que predictions: concat + groupby + normalización correcta
    if run_importances_list:
        print(f"🔬 Calculando importancias promedio para {len(run_importances_list)} seeds...")
        print(f"📋 Ejemplo importancias antes del promedio:")
        for i, df in enumerate(run_importances_list[:2]):  # Show first 2 for debugging
            print(f"   Seed {i+1}: {len(df)} features, importance range [{df['gini_importance'].min():.4f}, {df['gini_importance'].max():.4f}]")
        
        # MÉTODO CORRECTO: concat + groupby por feature name
        all_importances = pd.concat(run_importances_list, ignore_index=True)
        avg_importances = all_importances.groupby('feature')['gini_importance'].mean().reset_index()
        
        print(f"📊 Importancias promedio calculadas (antes de reescalado):")
        print(f"   📈 Total features: {len(avg_importances)}")
        print(f"   📊 Raw importance range: [{avg_importances['gini_importance'].min():.4f}, {avg_importances['gini_importance'].max():.4f}]")
        print(f"   🔝 Top 5 raw importances: {avg_importances.nlargest(5, 'gini_importance')['gini_importance'].tolist()}")
        
        # ✅ CORRECCIÓN: Normalizar por MÁXIMO (no por suma) para evitar valores > 1.0
        max_importance = avg_importances['gini_importance'].max()
        if max_importance > 0:
            # Normalizar a [0, 1] dividiendo por el máximo
            avg_importances['gini_importance'] = avg_importances['gini_importance'] / max_importance
        else:
            print("⚠️  Warning: All feature importances are 0!")
        
        # Sort by importance (descending)
        avg_importances = avg_importances.sort_values('gini_importance', ascending=False)
        
        print(f"✅ Importancias reescaladas:")
        print(f"   📊 Normalized range: [{avg_importances['gini_importance'].min():.4f}, {avg_importances['gini_importance'].max():.4f}]")
        print(f"   🎯 Features with importance > 0.5: {len(avg_importances[avg_importances['gini_importance'] > 0.5])}")
        print(f"   🔝 Top 10 normalized importances: {avg_importances['gini_importance'].head(10).tolist()}")
        
        print(f"Gini-based feature importances averaged across all seeds ({len(avg_importances)} features)")
        
        if args["neptune"]:
            avg_importances.to_csv("avg_feature_importances.csv", index=False)
            neptune_run["feature_importances/avg"].upload("avg_feature_importances.csv")
        else:
            avg_importances.to_csv("avg_feature_importances.csv", index=False)
            print("Gini-based feature importances saved to: avg_feature_importances.csv")
    else:
        print("No feature importances were extracted (classifier may not support it)")

    if args["neptune"]:
        avg_preds.to_csv("avg_probs.csv", index=False)
        neptune_run["predictions/avg"].upload("avg_probs.csv")
        neptune_run.stop()

    # Cleanup temporary files
    for f in ["preds_temp.csv", "avg_probs.csv", "avg_feature_importances.csv"]:
        if os.path.exists(f):
            os.remove(f)

