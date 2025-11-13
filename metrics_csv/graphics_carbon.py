import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- 1. CARGA DE DATOS ---
try:
    df = pd.read_csv('./csv_carbon/emissions_data copy.csv')
    print("Successfully loaded 'emissions_data copy.csv'.")
except FileNotFoundError:
    print("Error: 'emissions_data copy.csv' not found.")
    sys.exit()

# --- 2. PREPARACIÓN DE DATOS ---
# Limpieza de etiquetas de configuración
df['Clean_Config'] = (
    df['Configuration']
    .str.replace(' (Individual)', '', regex=False)
    .str.replace(' (Combined)', '', regex=False)
)
# Crear etiqueta final
df['Label'] = df['Method'] + ' (' + df['Clean_Config'] + ')'

# Cálculo del total
df['Total'] = df[['Training', 'Inference', 'PU_Processing', 'Fast-mRMR_Processing']].sum(axis=1)
df = df.sort_values('Total', ascending=True)

# --- 3. CONFIGURACIÓN DE ESTILO ---
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 9))

phases = ['Training', 'Inference', 'PU_Processing', 'Fast-mRMR_Processing']
colors = {
    'Training': '#6495ed',          # azul
    'Inference': '#a9a9a9',         # gris
    'PU_Processing': '#ff6347',     # rojo
    'Fast-mRMR_Processing': '#3cb371'  # verde
}

# --- 4. DIBUJAR BARRAS APILADAS ---
left = pd.Series([0.0] * len(df), index=df.index)
for phase in phases:
    values = df[phase]
    ax.barh(df['Label'], values, left=left, color=colors[phase], label=phase.replace('_', ' '))
    left += values

# --- 5. FORMATO FINAL ---
ax.set_xlabel('CO$_2$ Emissions (gCO$_2$e)', fontsize=16)
ax.set_title('CO$_2$ Emissions by Pipeline', fontsize=18)
ax.set_ylabel('')
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=12)

sns.despine(left=True, bottom=True)

# Leyenda
ax.legend(
    ncol=2,
    bbox_to_anchor=(0.5, -0.2),
    loc='upper center',
    frameon=False,
    title='Pipeline Phase',
    fontsize=12,
    title_fontsize=13
)

# --- 6. GUARDAR GRÁFICO ---
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('./carbon_img/co2_emissions_plot_narrow.png', dpi=300, bbox_inches='tight')

plt.show()
print("✅ Gráfico guardado con éxito como './carbon_img/co2_emissions_plot_narrow.png'")
