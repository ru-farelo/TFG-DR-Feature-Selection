# 🌱 Guía para Comparar Consumo CO2 entre Métodos

## 🎯 Objetivo

Comparar el consumo de CO2 de diferentes configuraciones del pipeline:

1. **Método original** (sin optimizaciones)
2. **Fast-MRMR** (selección rápida de características)
3. **PU Learning** (aprendizaje con ejemplos no etiquetados)
4. **Fast + PU** (combinación de ambos)

## 📊 Métricas Subidas a Neptune

### Estructura Simplificada

```
co2/
├── phases/
│   ├── training/
│   │   ├── mean_per_fold_g          # Promedio de CO2 por fold (eficiencia)
│   │   └── total_experiment_g       # Total del experimento (coste absoluto)
│   ├── inference/
│   ├── pu_preprocessing/            # Solo si usas PU Learning
│   ├── fast_mrmr/                   # Solo si usas Fast-MRMR
│   ├── bagging/                     # Solo si usas Bagging
│   └── hyperparameter_tuning/
├── total_experiment_g               # Total de todas las fases
└── total_experiment_kg              # Total en kilogramos
```

## 🚀 Cómo Ejecutar Experimentos para Comparar

### 1️⃣ Método Original (Baseline)

```powershell
python .\code\main.py `
    --dataset GO `
    --classifier BRF `
    --carbon_tracking `
    --neptune
```

**Fases medidas**: `training`, `inference`, `hyperparameter_tuning`

### 2️⃣ Método con Fast-MRMR

```powershell
python .\code\main.py `
    --dataset GO `
    --classifier BRF `
    --fast_mrmr `
    --fast_mrmr_k 50% `
    --carbon_tracking `
    --neptune
```

**Fases medidas**: `fast_mrmr`, `training`, `inference`, `hyperparameter_tuning`

### 3️⃣ Método con PU Learning

```powershell
python .\code\main.py `
    --dataset GO `
    --classifier BRF `
    --pu_learning similarity `
    --carbon_tracking `
    --neptune
```

**Fases medidas**: `pu_preprocessing`, `training`, `inference`, `hyperparameter_tuning`

### 4️⃣ Método Fast + PU

```powershell
python .\code\main.py `
    --dataset GO `
    --classifier BRF `
    --fast_mrmr `
    --fast_mrmr_k 50% `
    --pu_learning similarity `
    --carbon_tracking `
    --neptune
```

**Fases medidas**: `fast_mrmr`, `pu_preprocessing`, `training`, `inference`, `hyperparameter_tuning`

## 📈 Interpretación de Métricas

### `mean_per_fold_g` - Eficiencia por fold

**Pregunta**: ¿Cuánto CO2 consume en promedio cada fold?

**Ejemplo**:
```
training/mean_per_fold_g = 2.456g
```
Significa que cada fold de entrenamiento emite ~2.5g de CO2.

**Para comparar**:
- ✅ Menor = más eficiente
- ⚠️ Medir en la misma máquina y condiciones

### `total_experiment_g` - Coste absoluto total

**Pregunta**: ¿Cuánto CO2 costó toda la fase en el experimento completo?

**Ejemplo**:
```
training/total_experiment_g = 245.6g
```
Con 10 seeds × 10 folds = 100 entrenamientos, el total fue 245.6g.

**Para comparar**:
- ✅ Menor = menos coste absoluto
- 📊 Multiplicar por número de experimentos para estimar coste a largo plazo

## 🔍 Preguntas que Puedes Responder

### ¿Fast-MRMR reduce el coste de training?

Compara:
- Método Original: `training/mean_per_fold_g`
- Con Fast-MRMR: `training/mean_per_fold_g`

Si Fast-MRMR reduce features de 5000 → 500, el training debería ser más rápido y emitir menos.

**Pero también mira**:
- `fast_mrmr/total_experiment_g` - ¿El coste de selección compensa el ahorro?

### ¿PU Learning vale la pena en términos de CO2?

Compara:
- Método Original: `total_experiment_g`
- Con PU: `total_experiment_g`

El PU añade `pu_preprocessing` pero puede mejorar métricas. Pregunta:
- **¿El coste extra justifica la mejora en F1/AUC?**

### ¿Qué fase consume más?

Mira `co2/phases/*/total_experiment_g` y ordena de mayor a menor.

Típicamente:
1. `training` (mayor)
2. `hyperparameter_tuning` 
3. `pu_preprocessing` (si aplica)
4. `fast_mrmr` (si aplica)
5. `inference` (menor)

### ¿Cuál es el método más sostenible?

Compara `co2/total_experiment_kg` de los 4 métodos.

**Considera también**:
- Métricas de ML (F1, AUC, Precision, Recall)
- Trade-off: menor CO2 vs mejor rendimiento

## 📊 Ejemplo de Comparación

### Salida en Consola

```
======================================================================
🌱 CARBON EMISSIONS SUMMARY
======================================================================
  fast_mrmr                 |   0.124g/fold |     12.450g total
  pu_preprocessing          |   0.893g/fold |     89.320g total
  training                  |   1.456g/fold |    145.600g total
  inference                 |   0.023g/fold |      2.340g total
  hyperparameter_tuning     |   3.200g/fold |    320.000g total
----------------------------------------------------------------------
  TOTAL EXPERIMENT          |               |    569.710g (0.5697 kg)
======================================================================
```

### En Neptune

Navega a `co2/phases/` y compara entre runs:

| Método | training (g/fold) | total_experiment (g) |
|--------|-------------------|----------------------|
| Original | 2.456 | 245.6 |
| Fast-MRMR | 1.456 | 145.6 |
| PU Learning | 2.234 | 223.4 |
| Fast + PU | 1.389 | 138.9 |

**Conclusión**: Fast + PU es el más eficiente (menor CO2 por fold y menor coste total).

## 🎓 Buenas Prácticas

### Para comparaciones justas

1. ✅ **Misma máquina**: El hardware afecta las mediciones
2. ✅ **Mismos seeds**: Usa los mismos SEEDS para todos los métodos
3. ✅ **Mismas condiciones**: CPU load, temperatura ambiente, etc.
4. ✅ **Mismo dataset**: GO, PathDip, etc.
5. ✅ **Mismo clasificador**: BRF, CAT, XGB

### Para reportar en tu TFG

- **Tabla comparativa** de `mean_per_fold_g` por fase y método
- **Gráfico de barras** de `total_experiment_g` por método
- **Trade-off plot**: CO2 vs F1-Score (¿más sostenible = peor rendimiento?)
- **Porcentaje de reducción**: "Fast-MRMR reduce el coste en un 40.6%"

### Contextualizar las emisiones

- 1g de CO2 ≈ cargar un smartphone 1 vez
- 100g de CO2 ≈ conducir un coche 500 metros
- 1kg de CO2 ≈ volar 5 kilómetros en avión

Ejemplo: "Nuestro método Fast+PU emite 138.9g por experimento completo, equivalente a cargar ~139 smartphones."

## 🔧 Comandos Útiles

### Ejecutar los 4 métodos en secuencia

```powershell
# 1. Baseline
python .\code\main.py --dataset GO --classifier BRF --carbon_tracking --neptune

# 2. Fast-MRMR
python .\code\main.py --dataset GO --classifier BRF --fast_mrmr --fast_mrmr_k 50% --carbon_tracking --neptune

# 3. PU Learning
python .\code\main.py --dataset GO --classifier BRF --pu_learning similarity --carbon_tracking --neptune

# 4. Fast + PU
python .\code\main.py --dataset GO --classifier BRF --fast_mrmr --fast_mrmr_k 50% --pu_learning similarity --carbon_tracking --neptune
```

### Comparar en Neptune

1. Ve a tu proyecto Neptune
2. Navega a `co2/phases/training/mean_per_fold_g`
3. Compara runs con "Compare" view
4. Exporta a CSV o genera gráficos

## ✅ Checklist para tu TFG

- [ ] Ejecutar los 4 métodos con `--carbon_tracking`
- [ ] Recolectar datos de Neptune en tabla CSV
- [ ] Crear gráfico comparativo de consumo por fase
- [ ] Calcular porcentaje de reducción de CO2
- [ ] Comparar CO2 vs métricas de ML (F1, AUC)
- [ ] Contextualizar emisiones (smartphones, distancia en coche)
- [ ] Discutir trade-offs: eficiencia vs rendimiento
- [ ] Conclusión: ¿cuál es el método más sostenible?

¡Listo para demostrar que tu método no solo es mejor en métricas, sino también más sostenible! 🌱
