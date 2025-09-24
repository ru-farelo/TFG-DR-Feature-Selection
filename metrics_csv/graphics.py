import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Verify command line arguments
if len(sys.argv) < 2:
    print("Uso: python script.py <archivo_csv>")
    sys.exit(1)

# Obtain CSV file path from command line
csv_filename = sys.argv[1]
csv_path = os.path.join("csv", os.path.basename(csv_filename))

# Verify if the file exists
if not os.path.exists(csv_path):
    print(f"Error: El archivo {csv_path} no existe.")
    sys.exit(1)

# Load the CSV file
df = pd.read_csv(csv_path, sep=';')

# Convert the first column to numeric (assuming it's the percentage of features selected)
df["fast_mrmr_k"] = df["fast_mrmr_k"].str.replace("%", "").astype(float)

# Calculate differences from the 100% selection row
reference_row = df[df["fast_mrmr_k"] == 100].iloc[0]
differences = df.set_index("fast_mrmr_k").drop(100) - reference_row
differences["fast_mrmr_k"] = differences.index

# Identify the highest value in each metric (excluding 100%)
top_values = df.set_index("fast_mrmr_k").drop(100).idxmax()

# Create the figure
plt.figure(figsize=(12, 6))

# Dictionary of colors for lines
colors = plt.get_cmap("tab10").colors  # Predefined colors from matplotlib

# Plot each metric with its own color
for i, column in enumerate(df.columns[1:]):  # Exclude the first column (fast_mrmr_k)
    plt.plot(df["fast_mrmr_k"], df[column], marker='o', label=column, color=colors[i])
    
    # Identify and annotate the maximum point
    max_index = top_values[column]
    max_value = df.loc[df["fast_mrmr_k"] == max_index, column].values[0]

    # Anote the maximum value on the plot
    plt.annotate(f"{max_value:.3f}", 
                 (max_index, max_value), 
                 textcoords="offset points", 
                 xytext=(0, 5), 
                 ha='center', 
                 color=colors[i], 
                 fontsize=10, 
                 fontweight="bold")

    # Add the 100% values in red for comparison
    ref_value = reference_row[column]
    plt.annotate(f"{ref_value:.3f}", 
                 (100, ref_value), 
                 textcoords="offset points", 
                 xytext=(0, 5), 
                 ha='center', 
                 color='red', 
                 fontsize=10, 
                 fontweight="bold")

# Personalize the plot
plt.xlabel("Porcentaje de selección de características")
plt.ylabel("Valor de la métrica")
plt.title(f"Comparación de métricas - {os.path.basename(csv_filename).replace('.csv', '')}")
plt.legend()
plt.grid(True)

# Ajust x-ticks to show percentages
plt.xticks(df["fast_mrmr_k"], [f"{x}%" for x in df["fast_mrmr_k"]], rotation=90)
plt.subplots_adjust(bottom=0.2)  # Add space for x-ticks

# Save the plot
output_dir = "png"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f"{os.path.basename(csv_filename).replace('.csv', '')}.png")
plt.savefig(output_filename, bbox_inches="tight")  # Avoid clipping
print(f"Plot saved to {output_filename}")
