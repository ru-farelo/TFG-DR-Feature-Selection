# 🌱 Carbon Tracking con CodeCarbon

## Descripción

Este proyecto incluye instrumentación para medir el coste computacional (emisiones de CO2) de cada fase del pipeline de machine learning usando **CodeCarbon**.

### ✅ Características

- **Medición por fases**: bagging, fast_mrmr, pu_preprocessing, training, inference, hyperparameter_tuning
- **Solo proceso Python**: Usa `tracking_mode="process"` para medir únicamente el proceso Python, no todo el sistema
- **Granularidad múltiple**: 
  - Por fold individual
  - Media y total por run (seed)
  - Media y total por experimento completo (todos los seeds)
- **Integración con Neptune**: Subida automática de métricas de emisiones
- **Activación opcional**: Flag `--carbon_tracking` en config

## 📦 Instalación

```powershell
pip install codecarbon==2.5.0
```

> **Nota**: Ya está en `requirements.txt`, pero si trabajas en un entorno nuevo necesitas instalarlo explícitamente.

## 🚀 Uso

### Opción 1: Línea de comandos

```powershell
python .\code\main.py --dataset GO --classifier BRF --carbon_tracking
```

### Opción 2: Test rápido

Ejecuta el script de prueba con un experimento reducido:

```powershell
python .\test_carbon_tracking.py
```

Este script ejecuta 1 seed con el CV completo (10 folds por defecto) y muestra las emisiones por fase.

## 📊 Estructura de datos en Neptune

Las emisiones se organizan jerárquicamente:

```
emissions/
├── run_{seed}/                          # Detalle por fold
│   ├── bagging/
│   ├── fast_mrmr/
│   ├── pu_preprocessing/
│   ├── training/
│   ├── inference/
│   │   └── (valor en gramos por fold)
│   └── hyperparameter_tuning/
│       └── run_{seed}                   # Total del grid search
│
├── by_run/                              # Agregado por run
│   └── run_{seed}/
│       └── {phase}/
│           ├── avg_g                    # Media de 10 folds
│           └── total_g                  # Suma de 10 folds
│
└── summary/                             # 🎯 RESUMEN FINAL
    ├── {phase}/
    │   ├── mean_avg_g                   # Media de medias (todos los seeds)
    │   └── total_experiment_g           # Suma total del experimento
    ├── GRAND_TOTAL_g                    # Total absoluto
    └── GRAND_TOTAL_kg                   # Total en kilogramos
```

## 📈 Interpretación de métricas

### Por fold (emissions/run_{seed}/{phase})
Emisión en gramos de un fold específico de una fase específica.

### Por run (by_run/run_{seed}/{phase})
- **avg_g**: Media de los 10 folds para esa fase
- **total_g**: Suma de los 10 folds para esa fase

### Summary (emissions/summary)
- **mean_avg_g**: Media de las medias de todos los runs (promedio típico por fold)
- **total_experiment_g**: Suma total de todos los runs y todos los folds
- **GRAND_TOTAL_g**: Suma de todas las fases de todo el experimento

## 🔍 Ejemplo de salida

```
==================================================================
🌱 CARBON EMISSIONS SUMMARY (across all seeds)
==================================================================
  bagging                   | Avg per fold:   0.0523g | Total experiment:     5.2300g
  fast_mrmr                 | Avg per fold:   0.1245g | Total experiment:    12.4500g
  pu_preprocessing          | Avg per fold:   0.8932g | Total experiment:    89.3200g
  training                  | Avg per fold:   2.4567g | Total experiment:   245.6700g
  inference                 | Avg per fold:   0.0234g | Total experiment:     2.3400g
------------------------------------------------------------------
  TOTAL EXPERIMENT          |                      |    354.9100g (0.3549 kg)
==================================================================
```

## 🛠️ Configuración avanzada

En `code/carbon_utils.py` puedes modificar:

```python
tracker = EmissionsTracker(
    project_name=task_name,
    measure_power_secs=1,      # Frecuencia de medición (segundos)
    save_to_file=False,        # No guardar CSVs locales
    save_to_api=False,         # No enviar a API de CodeCarbon
    log_level="error",         # Nivel de logging
    tracking_mode="process"    # 🎯 Solo este proceso Python
)
```

### Modos de tracking disponibles

- `"process"` (recomendado): Solo el proceso Python actual
- `"machine"`: Toda la máquina (CPU, GPU, RAM global)

## ⚠️ Notas importantes

1. **Primera ejecución**: CodeCarbon puede tardar unos segundos en inicializar la primera vez
2. **Precisión**: Las emisiones son estimaciones basadas en TDP de hardware y mix energético regional
3. **Sin CodeCarbon**: Si no está instalado, el pipeline sigue funcionando (emisiones = 0g)
4. **Archivo temporales**: Se limpian automáticamente al final de la ejecución

## 🐛 Troubleshooting

### CodeCarbon no se instala
```powershell
pip install --upgrade pip
pip install codecarbon==2.5.0
```

### Emisiones siempre 0g
- Verifica que CodeCarbon esté instalado: `pip show codecarbon`
- Verifica que el tracking esté activado: `--carbon_tracking` en la línea de comandos
- Revisa la consola en busca de warnings de CodeCarbon

### Error "tracking_mode not supported"
Tu versión de CodeCarbon puede ser antigua. Actualiza a 2.5.0 o superior, o comenta la línea `tracking_mode="process"` en `carbon_utils.py`.

## 📚 Referencias

- [CodeCarbon Documentation](https://mlco2.github.io/codecarbon/)
- [Paper: Quantifying the Carbon Emissions of Machine Learning](https://arxiv.org/abs/1910.09700)
