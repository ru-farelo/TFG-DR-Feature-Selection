import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Data for the three methods across two feature set combinations
# Convert kg to grams (multiply by 1000) for better visualization
data = {
    'Method': ['Original', 'PU Learning', 'Fast-mRMR'],
    'PathDIP + CAT': [0.00153 * 1000, 0.00396 * 1000, 0.00379 * 1000],  # Convert kg to g
    'GO + CAT': [0.00128 * 1000, 0.00389 * 1000, 0.00330 * 1000]        # Convert kg to g
}

# Create a DataFrame from the data
df = pd.DataFrame(data).set_index('Method')

# Set up the bar chart
labels = df.columns
x = np.arange(len(labels))  # The label locations
width = 0.25  # The width of the bars

fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size for better visibility

# Define colors for each method using hexadecimal codes
colors = {
    'Original': '#8b6528', # Brown
    'PU Learning': '#9d69a8', # Purple
    'Fast-mRMR': '#6495ed' # A not-too-flashy blue
}

# Create the bars for each method
rects1 = ax.bar(x - width, df.loc['Original'], width, label='Original (non-PU)', color=colors['Original'])
rects2 = ax.bar(x, df.loc['PU Learning'], width, label='PU Learning', color=colors['PU Learning'])
rects3 = ax.bar(x + width, df.loc['Fast-mRMR'], width, label='Fast-mRMR (Propuesto)', color=colors['Fast-mRMR'])

# Add labels, title, and other customizations
ax.set_ylabel('Emisiones (g CO₂e)', fontsize=12)
ax.set_title('Emisiones de CO₂ por método de aprendizaje', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.6) # Add a horizontal grid for better readability

# Set y-axis limits to better show the differences
ax.set_ylim(0, max(df.max()) * 1.15)

# Function to add value labels on top of the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

# Add labels to all bars
add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

fig.tight_layout()

# Create carbon_img directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), '..', 'carbon_img')
os.makedirs(output_dir, exist_ok=True)

# Save the figure in multiple formats
output_path_png = os.path.join(output_dir, 'emisiones_co2_comparacion.png')
output_path_pdf = os.path.join(output_dir, 'emisiones_co2_comparacion.pdf')

plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.savefig(output_path_pdf, bbox_inches='tight')

print(f"📊 Gráfico guardado en:")
print(f"   PNG: {output_path_png}")
print(f"   PDF: {output_path_pdf}")
print(f"\n📈 Valores mostrados en gramos (g CO₂e) para mejor visualización")
print(f"   PathDIP + CAT - Original: {data['PathDIP + CAT'][0]:.2f} g")
print(f"   PathDIP + CAT - PU Learning: {data['PathDIP + CAT'][1]:.2f} g") 
print(f"   PathDIP + CAT - Fast-mRMR: {data['PathDIP + CAT'][2]:.2f} g")

plt.show()