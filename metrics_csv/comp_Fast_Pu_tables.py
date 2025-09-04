import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

COLUMN_MAP = {
    "F1-score": "f1",
    "G-Mean": "gmean",
    "AUC-ROC": "auc_roc",
    "AUC-PR": "auc_pr",
    "Recall@10": "recall_at_10",
    "NDCG": "ndcg_at_10",
    "Precision@10": "precision_at_10"
}

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

def extraer_fila_100(df):
    df["fast_mrmr_k"] = df["fast_mrmr_k"].astype(str).str.replace('%', '').astype(float)
    fila = df[df["fast_mrmr_k"] == 100]
    if fila.empty:
        fila = df.tail(1)
    fila = fila.iloc[0]
    resultados = {}
    for metrica, col in COLUMN_MAP.items():
        if col in df.columns:
            resultados[metrica] = f"{round(fila[col], 3)} (100.0%)"
        else:
            resultados[metrica] = ""
    return resultados

def generar_tabla_visual(df, titulo, output_path, color_max="#a1d99b", color_base="#f7f7f7", color_final="#fcbba1"):
    df = df.set_index("Configuración")
    plt.figure(figsize=(16, 3))
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
                if "Sin Selección" in df.index[i]:
                    cell.set_facecolor(color_final)
                elif col_values.iloc[i] == max_val:
                    cell.set_facecolor(color_max)
                else:
                    cell.set_facecolor(color_base)
        except:
            continue

    plt.title(titulo, fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Guardado: {output_path}")

def procesar_archivo(nombre_config, ruta):
    df = pd.read_csv(ruta, sep=";")
    resumen = {"Configuración": nombre_config}
    metricas = extraer_metricas_con_porcentaje(df)
    for metrica in COLUMN_MAP.keys():
        valor = metricas[metrica]["valor"]
        pct = metricas[metrica]["%"]
        resumen[metrica] = f"{valor} ({pct})" if valor is not None else ""
    return resumen

def main():
    parser = argparse.ArgumentParser(description="Genera comparativa de BRF y CAT con y sin PU Learning")
    parser.add_argument("--fast_BRF", required=True)
    parser.add_argument("--fast_CAT", required=True)
    parser.add_argument("--fast_BRF_PU", required=True)
    parser.add_argument("--fast_CAT_PU", required=True)
    parser.add_argument("--fast", required=True, help="Nombre visual para el método FAST")
    parser.add_argument("--pu", required=True, help="Nombre visual para el método FAST + PU")

    args = parser.parse_args()
    os.makedirs("./tables", exist_ok=True)

    configs = [
        (f"{args.fast} (BRF)", args.fast_BRF),
        (f"{args.fast} (CAT)", args.fast_CAT),
        (f"{args.pu} (BRF)", args.fast_BRF_PU),
        (f"{args.pu} (CAT)", args.fast_CAT_PU),
    ]

    fs_data = [procesar_archivo(nombre, ruta) for nombre, ruta in configs]

    # Cargar todos los CSV necesarios
    df_fast_brf = pd.read_csv(args.fast_BRF, sep=";")
    df_fast_cat = pd.read_csv(args.fast_CAT, sep=";")
    df_fast_brf_pu = pd.read_csv(args.fast_BRF_PU, sep=";")
    df_fast_cat_pu = pd.read_csv(args.fast_CAT_PU, sep=";")

    # Añadir las 4 filas de Sin Selección (100%)
    sin_fast_brf = {"Configuración": "Sin Selección BRF(100%)"}
    sin_fast_brf.update(extraer_fila_100(df_fast_brf))
    fs_data.append(sin_fast_brf)

    sin_fast_cat = {"Configuración": "Sin Selección CAT(100%)"}
    sin_fast_cat.update(extraer_fila_100(df_fast_cat))
    fs_data.append(sin_fast_cat)

    sin_pu_brf = {"Configuración": "Sin Selección BRF + PU(100%)"}
    sin_pu_brf.update(extraer_fila_100(df_fast_brf_pu))
    fs_data.append(sin_pu_brf)

    sin_pu_cat = {"Configuración": "Sin Selección CAT + PU(100%)"}
    sin_pu_cat.update(extraer_fila_100(df_fast_cat_pu))
    fs_data.append(sin_pu_cat)

    df_fs = pd.DataFrame(fs_data)

    visual_filename = f"Comparativa_{args.fast.replace(' ', '_')}_vs_{args.pu.replace(' ', '_')}.png"
    visual_title = (
        f"Comparativa de {args.fast} vs {args.pu}\n"
        f"Incluye configuraciones sin selección (100%) para BRF y CAT"
    )

    generar_tabla_visual(
        df_fs,
        visual_title,
        os.path.join("./tables", visual_filename)
    )

if __name__ == "__main__":
    main()
