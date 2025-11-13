import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATOS ---
models = {
    'Original (PathDIP)': {
        'training_cost': 0.633090, 'preprocessing_cost': 0, 'inference_cost': 0.020915
    },
    'PU Learning (PathDIP)': {
        'training_cost': 0.652734, 'preprocessing_cost': 0.037433, 'inference_cost': 0.022044
    },
    'Fast-mRMR (GO)': {
        'training_cost': 0.182752, 'preprocessing_cost': 3.619786, 'inference_cost': 0.011047
    },
    'Fast-mRMR (GO+PathDIP)': {
        'training_cost': 0.195448, 'preprocessing_cost': 4.037089, 'inference_cost': 0.012666
    }
}

# --- 2. CÁLCULOS ---
n_inferences = np.logspace(1, 5, num=100)
results = {}
for model_name, costs in models.items():
    initial_cost = costs['training_cost'] + costs['preprocessing_cost']
    results[model_name] = initial_cost + (n_inferences * costs['inference_cost'])

# --- 3. LÓGICA DEL GRÁFICO ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 6))

style_map = {
    'Original (PathDIP)':       {'color': 'darkgreen', 'linestyle': '-', 'linewidth': 2.5},
    'PU Learning (PathDIP)':    {'color': 'darkgreen', 'linestyle': '--', 'linewidth': 2.5},
    'Fast-mRMR (GO)':           {'color': 'darkgreen', 'linestyle': '-.', 'linewidth': 2.5},
    'Fast-mRMR (GO+PathDIP)':   {'color': 'limegreen', 'linestyle': '-.', 'linewidth': 2.5}
}

for model_name, cost_values in results.items():
    plt.plot(n_inferences, cost_values, label=model_name, **style_map[model_name])

# --- 4. FORMATEO Y GUARDADO ---
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="major", ls="--", alpha=0.5)

plt.xlabel('Number of Inferences', fontsize=14)
plt.ylabel('Total CO$_2$ Emissions (gCO$_2$e)', fontsize=14)
plt.title('Long-Term Cost Evolution', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)

# --- MODIFICACIÓN CLAVE: Anotaciones manuales para evitar superposición ---
# Dibuja las líneas verticales
genes_markers = [986, 1124, 50000]
for genes in genes_markers:
    plt.axvline(x=genes, color='gray', linestyle=':', alpha=0.7)

# Añade las etiquetas de texto a diferentes alturas
y_min, y_max = plt.ylim()
plt.text(genes_markers[0] * 1.1, y_min * 1.5, 'PathDIP genes', rotation=45, alpha=0.8, fontsize=10)
plt.text(genes_markers[1] * 1.1, y_min * 3.5, 'GO genes', rotation=45, alpha=0.8, fontsize=10) # <-- Posición Y más alta
plt.text(genes_markers[2] * 1.1, y_min * 1.5, 'Human genome (approx.)', rotation=45, alpha=0.8, fontsize=10)
# --- FIN DE LA MODIFICACIÓN ---

plt.legend(title='Method', fontsize=10)
plt.tight_layout()
plt.savefig('sustainability_analysis_final.png', dpi=300, bbox_inches='tight')

print("Gráfico 'sustainability_analysis_final.png' guardado con éxito.")