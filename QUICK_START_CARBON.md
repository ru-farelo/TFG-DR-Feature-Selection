# 🎯 Guía de Uso - Carbon Tracking

## ✅ Implementación Completada

Se ha implementado la medición de emisiones de CO2 con CodeCarbon con las siguientes características:

### 📊 Métricas Generadas

1. **Media por fase** (`mean_avg_g`): Promedio de emisiones por fold en cada fase
2. **Total del experimento** (`total_experiment_g`): Suma de todas las emisiones de esa fase
3. **Gran total** (`GRAND_TOTAL_g`): Suma de todas las fases y todos los seeds

### 🔧 Configuración

El tracking se activa con el flag `--carbon_tracking` en la línea de comandos.

## 🚀 Cómo Ejecutar

### 1️⃣ Instalar CodeCarbon (si no está instalado)

```powershell
pip install codecarbon==2.5.0
```

### 2️⃣ Ejecutar con tracking activado

```powershell
# Experimento completo con tracking
python .\code\main.py --dataset GO --classifier BRF --carbon_tracking

# Con PU Learning
python .\code\main.py --dataset GO --classifier BRF --pu_learning similarity --carbon_tracking

# Con Fast-MRMR y Bagging
python .\code\main.py --dataset GO --classifier BRF --fast_mrmr --fast_mrmr_k 50% --bagging --bagging_n 10% --carbon_tracking
```

### 3️⃣ Test rápido (1 seed, para validar)

```powershell
python .\test_carbon_tracking.py
```

## 📈 Salida Esperada

Al final de la ejecución verás un resumen como este:

```
======================================================================
🌱 CARBON EMISSIONS SUMMARY (across all seeds)
======================================================================
  training                  | Avg per fold:   2.4567g | Total experiment:   245.6700g
  inference                 | Avg per fold:   0.0234g | Total experiment:     2.3400g
  pu_preprocessing          | Avg per fold:   0.8932g | Total experiment:    89.3200g
  fast_mrmr                 | Avg per fold:   0.1245g | Total experiment:    12.4500g
  bagging                   | Avg per fold:   0.0523g | Total experiment:     5.2300g
----------------------------------------------------------------------
  TOTAL EXPERIMENT          |                      |    355.0100g (0.3550 kg)
======================================================================
```

## 🌍 Interpretación

- **mean_avg_g**: Cuánto CO2 emite en promedio un fold de esa fase
- **total_experiment_g**: Cuánto CO2 emitió esa fase en todo el experimento (10 seeds × 10 folds)
- **GRAND_TOTAL**: Emisión total del experimento completo

### Ejemplo de cálculo

Si `training/mean_avg_g = 2.4567g`:
- Cada fold de entrenamiento emite ~2.46g de CO2
- Con 10 folds × 10 seeds = 100 entrenamientos totales
- Total: ~245.67g para toda la fase de training

## 📊 Datos en Neptune

Si usas Neptune (`--neptune`), los datos se suben automáticamente:

```
emissions/
├── summary/                             # 🎯 Datos finales agregados
│   ├── training/
│   │   ├── mean_avg_g                   # Media de medias
│   │   └── total_experiment_g           # Suma total
│   ├── inference/
│   ├── pu_preprocessing/
│   ├── fast_mrmr/
│   ├── bagging/
│   ├── GRAND_TOTAL_g                    # Total absoluto
│   └── GRAND_TOTAL_kg                   # Total en kg
│
├── run_{seed}/{phase}/                  # Detalle por fold
└── by_run/run_{seed}/{phase}/           # Agregado por run
    ├── avg_g                            # Media de 10 folds
    └── total_g                          # Suma de 10 folds
```

## ⚙️ Configuración Avanzada

### Medir solo el proceso Python (por defecto)

El código está configurado para medir **solo el proceso Python actual**, no todo el sistema:

```python
tracking_mode="process"  # Solo este proceso
```

Si quieres medir toda la máquina:

```python
tracking_mode="machine"  # CPU, GPU, RAM global
```

Edita `code/carbon_utils.py` línea ~60.

### Frecuencia de medición

Por defecto mide cada 1 segundo:

```python
measure_power_secs=1
```

Puedes aumentar para reducir overhead (menos preciso):

```python
measure_power_secs=5  # Mide cada 5 segundos
```

## ❓ FAQ

### ¿Qué pasa si no instalo CodeCarbon?

El pipeline seguirá funcionando normalmente, pero las emisiones serán 0g. Verás un warning:

```
⚠️  CodeCarbon tracking requested but package not installed
   Install with: pip install codecarbon
```

### ¿Afecta al rendimiento?

CodeCarbon tiene overhead mínimo (~0.1-0.5% CPU). El tracking está optimizado para no interferir.

### ¿Puedo comparar fases?

¡Sí! Ese es el objetivo principal. Por ejemplo:

- **¿Qué consume más: training o hyperparameter tuning?**
- **¿Vale la pena el coste de PU Learning vs ganancia en métricas?**
- **¿Fast-MRMR reduce emisiones al reducir features?**

### ¿Cómo desactivo el tracking?

Simplemente no uses el flag `--carbon_tracking`:

```powershell
python .\code\main.py --dataset GO --classifier BRF
```

## 📚 Archivos Modificados

- ✅ `code/carbon_utils.py` - Wrapper robusto con control por config
- ✅ `code/config.py` - Ya tenía `--carbon_tracking`
- ✅ `code/model_training.py` - Instrumentación de fases
- ✅ `code/hyperparameter_tuning.py` - Medición del grid search
- ✅ `code/experiment.py` - Agregación por run
- ✅ `code/main.py` - Agregación final y subida a Neptune
- ✅ `code/neptune_utils.py` - Función upload_emissions_to_neptune
- ✅ `test_carbon_tracking.py` - Script de prueba
- ✅ `CARBON_TRACKING.md` - Documentación completa

## 🎓 Ejemplo Completo

```powershell
# 1. Asegurar que CodeCarbon está instalado
pip install codecarbon==2.5.0

# 2. Test rápido
python .\test_carbon_tracking.py

# 3. Experimento completo con tracking y Neptune
python .\code\main.py `
    --dataset GO `
    --classifier BRF `
    --pu_learning similarity `
    --fast_mrmr `
    --fast_mrmr_k 50% `
    --carbon_tracking `
    --neptune

# 4. Ver resultados en Neptune o en la consola
```

## ✨ Notas Finales

- Las emisiones son **estimaciones** basadas en TDP de hardware
- La precisión depende del mix energético de tu región
- CodeCarbon usa datos de [Electricity Maps](https://app.electricitymaps.com/)
- Para comparaciones relativas, los valores son muy fiables

¡Listo para medir el impacto ambiental de tu ML pipeline! 🌱
