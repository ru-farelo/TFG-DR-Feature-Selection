import pandas as pd
import os
import argparse
from functools import reduce

def intersect_genes_union_features(file_paths, output_dir="../data/"):
    if len(file_paths) < 2:
        print(" Se requieren al menos dos archivos CSV.")
        return

    # Read the files and capture the base names
    dfs = [pd.read_csv(path) for path in file_paths]
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]

    # Basic validation
    for i, df in enumerate(dfs):
        if 'Gene' not in df.columns:
            raise ValueError(f" El archivo '{file_paths[i]}' no contiene columna 'Gene'.")

    # Intersect genes
    gene_sets = [set(df['Gene']) for df in dfs]
    common_genes = reduce(lambda a, b: a & b, gene_sets)

    if not common_genes:
        print(" No hay genes comunes entre los archivos.")
        return

    # Filter DataFrames by common genes and remove duplicate 'Class' columns
    filtered_dfs = []
    for df in dfs:
        df = df[df['Gene'].isin(common_genes)].copy()
        class_cols = [col for col in df.columns if col.startswith('Class')]
        if len(class_cols) > 1:
            df.drop(columns=class_cols[:-1], inplace=True)
        filtered_dfs.append(df)

    # Make progressive merge by 'Gene'
    merged_df = filtered_dfs[0]
    for df in filtered_dfs[1:]:
        # Avoid duplication of 'Class' columns in the merge
        df_renamed = df.drop(columns=[col for col in df.columns if col == 'Class' and col in merged_df.columns])
        merged_df = pd.merge(merged_df, df_renamed, on='Gene', how='inner')

    # Ensure only one 'Class' column exists
    if 'Class' in merged_df.columns:
        class_col = merged_df.pop('Class')
        merged_df['Class'] = class_col

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "IntersectGenes_UnionFeatures_" + "_".join(base_names) + ".csv"
    output_path = os.path.join(output_dir, output_filename)
    merged_df.to_csv(output_path, index=False)

    print(f"\n Archivo guardado en: {output_path}")
    print(f" Genes comunes: {len(common_genes)}")
    print(f" Características totales combinadas: {merged_df.shape[1] - 2 if 'Class' in merged_df.columns else merged_df.shape[1] - 1}")

    return output_path

# Usage example:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intersect genes and merge features from multiple CSVs.")
    parser.add_argument("csvs", nargs="+", help="Archivos CSV a procesar.")
    parser.add_argument("--output_dir", default="../data/", help="Directorio de salida (opcional).")
    args = parser.parse_args()

    intersect_genes_union_features(args.csvs, output_dir=args.output_dir)
