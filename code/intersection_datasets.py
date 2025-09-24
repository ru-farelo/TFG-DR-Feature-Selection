import pandas as pd
import sys
import os
from functools import reduce

def intersect_genes_and_export(file_paths, output_dir="../data/"):
    if len(file_paths) < 2:
        raise ValueError("Se requieren al menos dos archivos para hacer la intersección.")

    # Read the datasets and capture the base names
    dataframes = []
    gene_sets = []
    for path in file_paths:
        df = pd.read_csv(path)
        gene_column = df.columns[0]
        name = os.path.splitext(os.path.basename(path))[0]
        dataframes.append((name, df, gene_column))
        gene_sets.append(set(df[gene_column]))

    # Calculate common genes
    common_genes = reduce(lambda a, b: a & b, gene_sets)

    if not common_genes:
        print(" No se encontraron genes comunes.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, (name, df, gene_column) in enumerate(dataframes):
        # Create name with the rest of datasets
        other_names = [n for j, (n, _, _) in enumerate(dataframes) if j != i]
        output_name = f"{name}_Common_Genes_with_{'_'.join(other_names)}.csv"
        output_path = os.path.join(output_dir, output_name)

        # Filter only common genes
        filtered_df = df[df[gene_column].isin(common_genes)].copy()

        # Save CSV
        filtered_df.to_csv(output_path, index=False)
        print(f" Guardado: {output_path} ({filtered_df.shape[0]} genes)")

    print(f"\n Total de genes comunes encontrados: {len(common_genes)}")

# Usage example:
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python intersect_genes_export.py <dataset1.csv> <dataset2.csv> ...")
    else:
        file_paths = sys.argv[1:]
        intersect_genes_and_export(file_paths)
