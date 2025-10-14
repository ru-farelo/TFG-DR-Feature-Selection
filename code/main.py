import os
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from code.config import read_config
from code.experiment import run_experiment

# Optional import for Neptune
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("⚠️  Neptune no disponible. Ejecutando sin logging en Neptune.")

if __name__ == "__main__":

    # 🌱 Initialize CO2 tracker with task-based measurement
    tracker = EmissionsTracker(
        tracking_mode="process", 
        save_to_file=False, 
        log_level="ERROR"
    )
    tracker.start()

    SEEDS = [14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903]

    args = read_config()

    if args["neptune"] and NEPTUNE_AVAILABLE:
        neptune_run = neptune.init_run(
            project="farelo/Restriccion-Dietetica",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNjQwYzdjNi0yNmQ5LTQzZmQtODIxMy01YmJkZDEzMTA0NWEifQ==",
        )
        neptune_run["parameters"] = neptune.utils.stringify_unsupported(args)
    elif args["neptune"] and not NEPTUNE_AVAILABLE:
        print("⚠️  Neptune solicitado pero no disponible. Continuando sin Neptune.")
        neptune_run = None
    else:
        neptune_run = None
    run_metrics_list = []
    run_preds_list = []
    run_importances_list = []  # Store feature importances from each run
    run_emissions_list = []    # Store carbon emissions from each run
    search_space = {
        "pu_k": args["pu_k"],
        "pu_t": args["pu_t"],
    }

    for seed in SEEDS:
        run_metrics, run_preds, run_importances, cv_emissions = run_experiment(
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
            tracker=tracker  # Pass CO2 tracker for task-based measurement
        )
        
        run_metrics_list.append(run_metrics)
        run_preds_list.append(run_preds)
        
        # Only track emissions if carbon tracking is enabled
        if cv_emissions is not None:
            run_emissions_list.append({"seed": seed, "emissions_kg": cv_emissions})
            print(f"🌱 Seed {seed} completed with CO2 tracking")
        else:
            print(f"Seed {seed} completed (CO2 tracking disabled)")
        
        if run_importances is not None:
            run_importances['seed'] = seed
            run_importances_list.append(run_importances)

    for metric in run_metrics_list[0][0].keys():
        avg = np.mean([np.mean([fold[metric] for fold in run_metrics]) for run_metrics in run_metrics_list])
        if args["neptune"]:
            neptune_run[f"metrics/avg/test/{metric}"] = avg
        else:
            print(f"metrics/avg/test/{metric}: {avg}")

    # Average predictions - CORRECTED: Group by gene index, not DataFrame position
    # Concatenate all predictions and group by gene to calculate proper average
    print(f"📊 Calculando predicciones promedio para {len(run_preds_list)} seeds...")
    print(f"📋 Ejemplo DataFrames antes del promedio:")
    for i, df in enumerate(run_preds_list[:2]):  # Show first 2 for debugging
        print(f"   Seed {i+1}: {len(df)} genes, prob range [{df['prob'].min():.4f}, {df['prob'].max():.4f}]")
    
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

    # Average feature importances across seeds (Gini-based importance from ensemble)
    # Following the reference code pattern: concat + groupby + rescale
    if run_importances_list:
        print(f"🔬 Calculando importancias promedio para {len(run_importances_list)} seeds...")
        print(f"📋 Ejemplo importancias antes del promedio:")
        for i, df in enumerate(run_importances_list[:2]):  # Show first 2 for debugging
            print(f"   Seed {i+1}: {len(df)} features, importance range [{df['gini_importance'].min():.4f}, {df['gini_importance'].max():.4f}]")
        
        # REFERENCE PATTERN: Concatenate all the feature importances
        all_importances = pd.concat(run_importances_list, ignore_index=True)
        
        # REFERENCE PATTERN: Group by feature name and compute the average importance  
        avg_importances = all_importances.groupby('feature')['gini_importance'].mean().reset_index()
        
        print(f"📊 Importancias promedio calculadas (antes de reescalado):")
        print(f"   📈 Total features: {len(avg_importances)}")
        print(f"   📊 Raw importance range: [{avg_importances['gini_importance'].min():.4f}, {avg_importances['gini_importance'].max():.4f}]")
        print(f"   🔝 Top 5 raw importances: {avg_importances.nlargest(5, 'gini_importance')['gini_importance'].tolist()}")
        
        # REFERENCE PATTERN: Rescale importances to [0, 100] (or [0, 1] * 100)
        max_importance = avg_importances['gini_importance'].max()
        if max_importance > 0:
            # Option 1: Scale to [0, 100] like reference
            avg_importances['importance_scaled_100'] = 100 * avg_importances['gini_importance'] / max_importance
            # Option 2: Keep original gini scale [0, 1] but normalize
            avg_importances['gini_importance'] = avg_importances['gini_importance'] / max_importance
        else:
            print("⚠️  Warning: All feature importances are 0!")
            avg_importances['importance_scaled_100'] = 0
        
        # Sort by importance (descending)
        avg_importances = avg_importances.sort_values('gini_importance', ascending=False)
        
        print(f"✅ Importancias reescaladas:")
        print(f"   📊 Normalized range: [{avg_importances['gini_importance'].min():.4f}, {avg_importances['gini_importance'].max():.4f}]")
        print(f"   📊 Scaled [0-100] range: [{avg_importances['importance_scaled_100'].min():.1f}, {avg_importances['importance_scaled_100'].max():.1f}]")
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

    # Final CO2 tracking summary with task-based breakdown
    tracker.stop()
    
    # Get task-specific emissions
    task_emissions = getattr(tracker, '_task_emissions', {})
    total_emissions = tracker.final_emissions
    
    print(f"\n🌱 === CO2 FOOTPRINT (Task-based breakdown) ===")
    print(f"📊 Total emissions: {total_emissions * 1000:.2f} g CO2e")
    print(f"📊 Average per seed: {total_emissions * 1000 / len(SEEDS):.2f} g CO2e")
    
    # Show breakdown by task with proper phase names
    if task_emissions:
        print(f"\n🔍 Task breakdown:")
        total_task_emissions = sum(task_emissions.values())
        
        for task_name, emissions in task_emissions.items():
            percentage = (emissions / total_task_emissions * 100) if total_task_emissions > 0 else 0
            
            # Display proper names based on what was measured
            if task_name == "feature_selection":
                phase_name = "🔬 Feature Selection (Fast-mRMR)"
            elif task_name == "instance_selection":
                phase_name = "🔬 Instance Selection (PU Learning)"
            elif task_name == "training":
                phase_name = "🚀 Model Training"
            elif task_name == "inference":
                phase_name = "🎯 Inference/Prediction"
            else:
                phase_name = f"🔧 {task_name}"
                
            print(f"   {phase_name}: {emissions * 1000:.2f} g CO2e ({percentage:.1f}%)")
    
    # Show what phases were actually measured
    phases_measured = []
    if args["fast_mrmr"]:
        phases_measured.append("Feature Selection (Fast-mRMR)")
    elif args["pu_learning"]:
        phases_measured.append("Instance Selection (PU Learning)")
    phases_measured.extend(["Training", "Inference"])
    
    print(f"📊 Measured phases: {' + '.join(phases_measured)}")

    if args["neptune"]:
        avg_preds.to_csv("avg_probs.csv", index=False)
        neptune_run["predictions/avg"].upload("avg_probs.csv")
        neptune_run["carbon_footprint/total_emissions_g"] = total_emissions * 1000
        neptune_run["carbon_footprint/avg_per_seed_g"] = total_emissions * 1000 / len(SEEDS)
        neptune_run["carbon_footprint/methodology"] = "task_based_measurement"
        
        # Upload task-specific emissions
        if task_emissions:
            for task_name, emissions in task_emissions.items():
                neptune_run[f"carbon_footprint/tasks/{task_name}_g"] = emissions * 1000
                
        neptune_run.stop()

    # Cleanup temporary files
    for f in ["preds_temp.csv", "avg_probs.csv", "avg_feature_importances.csv"]:
        if os.path.exists(f):
            os.remove(f)
