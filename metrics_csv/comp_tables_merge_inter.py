import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Mapeo de métricas a columnas reales
COLUMN_MAP = {
    "F1-score": "f1",
    "G-Mean": "gmean",
    "AUC-ROC": "auc_roc",
    "AUC-PR": "auc_pr",
    "Recall@10": "recall_at_10",
    "NDCG": "ndcg_at_10",
    "Precision@10": "precision_at_10"
}

# Renombrar para mostrar en tabla con claridad
def renombrar_config(nombre):
    if nombre == "InterGOPathDip_BRF":
        return "Union (GO+PathDip) - BRF"
    if nombre == "InterGOPathDip_CAT":
        return "Union (GO+PathDip) - CAT"
    if "GO_Common_Genes_PathDip_BRF" in nombre:
        return "GO only - BRF"
    if "PathDip_Common_Genes_GO_BRF" in nombre:
        return "PathDip only - BRF"
    if "GO_Common_Genes_PathDip_CAT" in nombre:
        return "GO only - CAT"
    if "PathDip_Common_Genes_GO_CAT" in nombre:
        return "PathDip only - CAT"
    return nombre

def extraer_metricas_con_porcentaje(df):
    resultados = {}
    if "fast_mrmr_k" in df.columns:
        df["fast_mrmr_k"] = df["fast_mrmr_k"].astype(str).str.replace('%', '').astype(float)

    for metrica, col in COLUMN_MAP.items():
        if col in df.columns:
            max_idx = df[col].idxmax()
            max_val = df[col].max()
            max_pct = df.loc[max_idx, "fast_mrmr_k"]
            resultados[metrica] = {
                "valor": round(max_val, 3),
                "%": f"{max_pct:.1f}%"
            }
        else:
            resultados[metrica] = {"valor": None, "%": ""}
    return resultados

def generar_tabla_visual(df, titulo, output_path, color_max="#a1d99b", color_base="#f7f7f7"):
    df = df.set_index("Configuración")
    plt.figure(figsize=(18, 0.6 * len(df) + 1))
    plt.axis('off')
    tabla = plt.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colLoc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.4)

    for j in range(len(df.columns)):
        try:
            col_values = df.iloc[:, j].apply(lambda x: float(str(x).split()[0]) if pd.notnull(x) else None)
            max_val = col_values.max()
            for i in range(len(df)):
                cell = tabla[i + 1, j]
                if col_values.iloc[i] == max_val:
                    cell.set_facecolor(color_max)
                else:
                    cell.set_facecolor(color_base)
        except:
            continue

    plt.title(titulo, fontsize=13, weight='bold')
    plt.subplots_adjust(top=0.75)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Guardado: {output_path}")

def procesar_dataframe(nombre_config, df):
    resumen = {"Configuración": nombre_config}
    metricas = extraer_metricas_con_porcentaje(df)
    for metrica in COLUMN_MAP.keys():
        valor = metricas[metrica]["valor"]
        pct = metricas[metrica]["%"]
        resumen[metrica] = f"{valor} ({pct})" if valor is not None else ""
    return resumen

def main():
    parser = argparse.ArgumentParser(description="Genera tabla comparativa BRF vs CAT con distintas configuraciones de features")
    parser.add_argument("--brf_union_features", required=True, help="CSV de BRF con genes intersectados + unión de características")
    parser.add_argument("--brf_single_features", nargs='+', required=True, help="Uno o más CSVs de BRF con genes intersectados + características por conjunto")
    parser.add_argument("--cat_union_features", required=True, help="CSV de CAT con genes intersectados + unión de características")
    parser.add_argument("--cat_single_features", nargs='+', required=True, help="Uno o más CSVs de CAT con genes intersectados + características por conjunto")
    parser.add_argument("--union_name", required=True, help="Nombre para la configuración de unión de características")
    parser.add_argument("--single_name", required=True, help="Nombre para la configuración con características específicas por conjunto")

    args = parser.parse_args()
    os.makedirs("./tables", exist_ok=True)

    configs = []

    # Unión
    configs.append((f"{args.union_name}_BRF", pd.read_csv(args.brf_union_features, sep=";")))
    configs.append((f"{args.union_name}_CAT", pd.read_csv(args.cat_union_features, sep=";")))

    # Individuales BRF
    for path in args.brf_single_features:
        name = os.path.splitext(os.path.basename(path))[0]
        configs.append((name, pd.read_csv(path, sep=";")))

    # Individuales CAT
    for path in args.cat_single_features:
        name = os.path.splitext(os.path.basename(path))[0]
        configs.append((name, pd.read_csv(path, sep=";")))

    fs_data = []
    for nombre_config, df in configs:
        resumen = procesar_dataframe(renombrar_config(nombre_config), df)
        fs_data.append(resumen)

    df_fs = pd.DataFrame(fs_data)

    visual_filename = f"{args.union_name}_vs_{args.single_name}_Detallado_Comparativa_BRF_CAT.png"
    visual_title = (
        f"Comparativa de clasificadores BRF y CAT sobre configuraciones con:\n"
        f"- Unión (GO + PathDip): genes en común + TODAS las características combinadas\n"
        f"- GO only / PathDip only: genes en común + características específicas por conjunto"
    )

    generar_tabla_visual(
        df_fs,
        visual_title,
        os.path.join("./tables", visual_filename)
    )

if __name__ == "__main__":
    main()
