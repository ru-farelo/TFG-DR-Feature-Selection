import pandas as pd
import sys
import os
from functools import reduce

def intersect_genes_and_export(file_paths, output_dir="../data/"):
    if len(file_paths) < 2:
        raise ValueError("Se requieren al menos dos archivos para hacer la intersección.")

    # Leer los datasets y capturar los nombres base
    dataframes = []
    gene_sets = []
    for path in file_paths:
        df = pd.read_csv(path)
        gene_column = df.columns[0]
        name = os.path.splitext(os.path.basename(path))[0]
        dataframes.append((name, df, gene_column))
        gene_sets.append(set(df[gene_column]))

    # Calcular genes comunes
    common_genes = reduce(lambda a, b: a & b, gene_sets)

    if not common_genes:
        print(" No se encontraron genes comunes.")
        return

    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    for i, (name, df, gene_column) in enumerate(dataframes):
        # Crear nombre con el resto de datasets
        other_names = [n for j, (n, _, _) in enumerate(dataframes) if j != i]
        output_name = f"{name}_Common_Genes_with_{'_'.join(other_names)}.csv"
        output_path = os.path.join(output_dir, output_name)

        # Filtrar solo genes comunes
        filtered_df = df[df[gene_column].isin(common_genes)].copy()

        # Guardar CSV
        filtered_df.to_csv(output_path, index=False)
        print(f" Guardado: {output_path} ({filtered_df.shape[0]} genes)")

    print(f"\n Total de genes comunes encontrados: {len(common_genes)}")

# Uso desde terminal
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python intersect_genes_export.py <dataset1.csv> <dataset2.csv> ...")
    else:
        file_paths = sys.argv[1:]
        intersect_genes_and_export(file_paths)
