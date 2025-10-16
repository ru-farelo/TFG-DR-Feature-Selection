"""
Script de prueba para validar la medición de emisiones con CodeCarbon.
Ejecuta un experimento reducido (1 seed, 2 folds) para verificar funcionamiento.
"""
import sys
sys.path.insert(0, '.')

from code.config import read_config
from code.experiment import run_experiment
import code.carbon_utils as carbon

# Simular args mínimos
test_args = {
    "dataset": "GO",  # Cambiar por tu dataset
    "classifier": "BRF",
    "pu_learning": False,
    "fast_mrmr": False,
    "fast_mrmr_k": 0,
    "bagging": False,
    "bagging_n": 0,
    "bagging_groups": 5,
    "carbon_tracking": True,  # 🌱 ACTIVAR TRACKING
    "neptune": False,  # Sin Neptune para test rápido
}

# Activar tracking
carbon.set_tracking_enabled(True)

if carbon.has_codecarbon():
    print("✅ CodeCarbon disponible - tracking ACTIVADO")
else:
    print("⚠️  CodeCarbon NO disponible - emisiones serán 0g")
    print("   Instala con: pip install codecarbon")

print("\n🧪 Ejecutando experimento de prueba...")
print("   Dataset:", test_args["dataset"])
print("   Classifier:", test_args["classifier"])
print("   Seeds: [42] (solo 1 para test)")
print("   Folds: Usa el CV configurado en experiment.py\n")

# Ejecutar con 1 seed
seed = 42
search_space = {"pu_k": 10, "pu_t": None}

run_metrics, run_preds, run_importances, run_emissions = run_experiment(
    dataset=test_args["dataset"],
    classifier=test_args["classifier"],
    pu_learning=test_args["pu_learning"],
    fast_mrmr=test_args["fast_mrmr"],
    fast_mrmr_percentage=test_args["fast_mrmr_k"],
    search_space=search_space,
    random_state=seed,
    neptune_run=None,
    bagging=test_args["bagging"],
    bagging_n=test_args["bagging_n"],
    bagging_groups=test_args["bagging_groups"],
    extract_importances=True,
)

print("\n" + "="*70)
print("🌱 RESULTADOS DEL TEST - EMISIONES")
print("="*70)

if run_emissions:
    for phase, stats in run_emissions.items():
        print(f"  {phase:25s} | {stats['avg_g']:7.3f}g/fold | {stats['total_g']:10.3f}g total")
    
    total_g = sum(stats['total_g'] for stats in run_emissions.values())
    print("-"*70)
    print(f"  {'TOTAL':25s} |               | {total_g:10.3f}g ({total_g/1000:.4f} kg)")
else:
    print("  ⚠️  No se recolectaron emisiones (¿CodeCarbon instalado?)")

print("="*70)
print("\n✅ Test completado!")
print("   Si ves emisiones > 0g, la instrumentación funciona correctamente.")
print("   Para comparar métodos, ejecuta los 4 experimentos con --carbon_tracking")
print("\n📖 Lee CO2_COMPARISON_GUIDE.md para saber cómo comparar métodos")
