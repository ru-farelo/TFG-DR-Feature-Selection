import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Verificar que se proporciona un archivo CSV
if len(sys.argv) < 2:
    print("Uso: python script.py <archivo_csv>")
    sys.exit(1)

# Obtener solo el nombre del archivo sin la ruta completa
csv_filename = sys.argv[1]
csv_path = os.path.join("csv", os.path.basename(csv_filename))

# Verificar si el archivo CSV existe
if not os.path.exists(csv_path):
    print(f"Error: El archivo {csv_path} no existe.")
    sys.exit(1)

# Cargar los datos desde el CSV
df = pd.read_csv(csv_path, sep=';')

# Convertir el porcentaje a números para la gráfica
df["fast_mrmr_k"] = df["fast_mrmr_k"].str.replace("%", "").astype(float)

# Calcular la diferencia con respecto al 100%
reference_row = df[df["fast_mrmr_k"] == 100].iloc[0]
differences = df.set_index("fast_mrmr_k").drop(100) - reference_row
differences["fast_mrmr_k"] = differences.index

# Identificar el mayor valor en cada métrica (sin contar el 100%)
top_values = df.set_index("fast_mrmr_k").drop(100).idxmax()

# Crear la figura
plt.figure(figsize=(12, 6))

# Diccionario de colores para líneas
colors = plt.get_cmap("tab10").colors  # Colores predefinidos de matplotlib

# Graficar cada métrica con su propio color
for i, column in enumerate(df.columns[1:]):  # Excluir la primera columna (fast_mrmr_k)
    plt.plot(df["fast_mrmr_k"], df[column], marker='o', label=column, color=colors[i])
    
    # Identificar el punto máximo
    max_index = top_values[column]
    max_value = df.loc[df["fast_mrmr_k"] == max_index, column].values[0]

    # Anotación en el color de la línea
    plt.annotate(f"{max_value:.3f}", 
                 (max_index, max_value), 
                 textcoords="offset points", 
                 xytext=(0, 5), 
                 ha='center', 
                 color=colors[i], 
                 fontsize=10, 
                 fontweight="bold")

    # Agregar los valores de 100% en rojo para comparación
    ref_value = reference_row[column]
    plt.annotate(f"{ref_value:.3f}", 
                 (100, ref_value), 
                 textcoords="offset points", 
                 xytext=(0, 5), 
                 ha='center', 
                 color='red', 
                 fontsize=10, 
                 fontweight="bold")

# Personalizar la gráfica
plt.xlabel("Porcentaje de selección de características")
plt.ylabel("Valor de la métrica")
plt.title(f"Comparación de métricas - {os.path.basename(csv_filename).replace('.csv', '')}")
plt.legend()
plt.grid(True)

# Ajustar los valores del eje X para mejorar espaciado
plt.xticks(df["fast_mrmr_k"], [f"{x}%" for x in df["fast_mrmr_k"]], rotation=90)
plt.subplots_adjust(bottom=0.2)  # Agregar margen inferior

# Guardar la gráfica en el directorio png
output_dir = "png"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f"{os.path.basename(csv_filename).replace('.csv', '')}.png")
plt.savefig(output_filename, bbox_inches="tight")  # Evita recortes
print(f"Gráfica guardada en {output_filename}")
