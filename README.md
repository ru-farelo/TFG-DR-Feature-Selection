# Trabajo de Fin de Grado: Selección de Características en Datos Biomédicos mediante Métodos de Aprendizaje Automático

## Descripción General

Este repositorio contiene el código y los recursos desarrollados para el Trabajo de Fin de Grado titulado **"Selección de Características en Datos Biomédicos mediante Métodos de Aprendizaje Automático"**. El objetivo principal es investigar y aplicar técnicas de selección de características sobre conjuntos de datos biomédicos, evaluando el impacto en el rendimiento de modelos de clasificación y la interpretación biológica de los resultados.

El proyecto integra métodos estadísticos, algoritmos de aprendizaje automático y procesamiento de datos biomédicos, con especial énfasis en la comparación de diferentes enfoques de selección de características y su aplicación en datos de expresión génica y anotaciones funcionales.

## Estructura del Proyecto

- `code/`: Scripts principales del proyecto, incluyendo procesamiento de datos, entrenamiento de modelos, selección de características y evaluación.
- `data/`: Conjuntos de datos utilizados en los experimentos, en formato CSV.
- `metrics_csv/`: Resultados experimentales y métricas generadas durante la ejecución de los experimentos.
- `plib/` y `src_c/`: Implementaciones en C++ de algoritmos de selección de características y utilidades de procesamiento.
- `utils/`: Herramientas adicionales para generación y lectura de datos.
- `requirements.txt`: Dependencias necesarias para ejecutar el entorno Python del proyecto.

## Instalación

1. Clonar el repositorio:
	```powershell
	git clone https://github.com/ru-farelo/TFG-DR-Feature-Selection.git
	```
2. Instalar las dependencias de Python:
	```powershell
	pip install -r requirements.txt
	```
3. (Opcional) Compilar los módulos en C++ si se requiere:
	```powershell
	cd src_c
	make
	```

## Uso

Los principales scripts se encuentran en el directorio `code/`. El flujo típico de trabajo incluye:

1. **Procesamiento de datos**: Preprocesamiento y discretización de los datos biomédicos.
2. **Selección de características**: Ejecución de algoritmos de selección (por ejemplo, FastMRMR, Bagging Disjoint).
3. **Entrenamiento y evaluación de modelos**: Entrenamiento de modelos de clasificación y cálculo de métricas.
4. **Análisis de resultados**: Generación de tablas y gráficos para la interpretación de los resultados.

Ejemplo de ejecución principal:
```powershell
python code/main.py
```

## Cómo probar el código

El sistema permite configurar y ejecutar los experimentos mediante argumentos en línea de comandos. El script principal (`main.py`) acepta múltiples parámetros que controlan el flujo de trabajo, la selección de características, el tipo de clasificador, la validación cruzada y otras opciones avanzadas.

A continuación se muestran ejemplos de ejecución con distintas combinaciones de argumentos:

- **Ejemplo básico sin selección**:
	```powershell
	python code/main.py --dataset PathDip --classifier BRF
	```

- **Selección de características con Fast MRMR**:
	```powershell
	python code/main.py --dataset PathDip --classifier BRF --fast_mrmr --fast_mrmr_k 5%
	```

- **Aprendizaje PU (Positive-Unlabeled)**:
	```powershell
	python code/main.py --dataset PathDip --classifier BRF --pu_learning True --pu_k 10 --pu_t 0.1
	```

- **Combinación de Fast MRMR y Bagging**:
	```powershell
	python code/main.py --dataset PathDip --classifier BRF --fast_mrmr --fast_mrmr_k 10% --bagging --bagging_n 5% --bagging_groups 5
	```

- **Combinación de Bagging, Fast MRMR y Aprendizaje PU**:
	```powershell
	python code/main.py --dataset PathDip --classifier BRF --fast_mrmr --fast_mrmr_k 10% --bagging --bagging_n 5% --bagging_groups 5 --pu_learning True --pu_k 10 --pu_t 0.1
	```

- **Activación de logging con Neptune**:
	```powershell
	python code/main.py --dataset GtexDataset --classifier BRF --neptune True
	```

Cada argumento está documentado en el código fuente (`config.py`) y permite adaptar el comportamiento del sistema a los requisitos de cada experimento. Esta flexibilidad facilita la comparación de métodos y la reproducibilidad de los resultados.

## Dependencias

Las principales librerías utilizadas incluyen:
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `neptune`
- `codecarbon` - Para medición de huella de carbono
- Otras especificadas en `requirements.txt`

## Medición de Huella de Carbono

El proyecto integra [CodeCarbon](https://github.com/mlco2/codecarbon) para medir automáticamente las emisiones de CO2 durante la ejecución de los experimentos. Esta funcionalidad permite evaluar el impacto ambiental de diferentes configuraciones del pipeline:

- **Sin selección de características**: Medición del coste computacional completo
- **Con selección de características**: Comparación del ahorro computacional y emisiones

Los resultados se guardan en:
- `./carbon_emissions/emissions.csv`: Reporte detallado de emisiones
- Neptune (si está habilitado): Métricas de carbono en `carbon_footprint/`

Ejemplo de salida:
```
🌱 Total CO2 emissions: 0.012345 kg CO2
📊 Detailed report saved in: ./carbon_emissions/emissions.csv
```

## Datos

Los archivos en `data/` contienen los conjuntos de datos biomédicos empleados en los experimentos, incluyendo datos de expresión génica, anotaciones GO, PathDip, KEGG, PPI, entre otros.

## Resultados

Los resultados experimentales se almacenan en `metrics_csv/`, incluyendo métricas de rendimiento, tablas comparativas y gráficos generados durante el análisis.

## Documentación

Para una descripción detallada de la metodología, los experimentos realizados y la interpretación de los resultados, consultar el documento PDF adjunto:  
`TFG_FIC_UDC_Ruben.pdf`

## Autor

Rubén Farelo  
Facultad de Informática, Universidade da Coruña  
Curso 2024/2025

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

