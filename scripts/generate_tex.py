#!/usr/bin/env python3
"""
generate_tex.py — Genera el código fuente LaTeX (report.tex) unificado
a partir de los resultados en benchmark_results.csv
"""
import os
import sys
import pandas as pd

def escape_tex(text):
    return text.replace('_', r'\_').replace('%', r'\%').replace('γ', r'$\gamma$')

def generate_report(csv_path="benchmark_results.csv", out_path="report.tex"):
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    datasets = df["dataset"].unique()
    classifiers = df["classifier"].unique()

    # --- TABLA UNIFICADA (ACC, F1, TIME) ---
    table_tex = "\\begin{table*}[h!]\n\\centering\n"
    table_tex += "\\caption{Resultados Consolidados: Accuracy, F1-Score y Tiempo (ms) por Dataset y Clasificador}\n"
    table_tex += "\\label{tab:unified}\n"
    table_tex += "\\vspace{0.2cm}\n"
    table_tex += "\\resizebox{\\textwidth}{!}{\n"
    
    # 1 column for dataset + 3 columns per classifier
    col_format = "l" + "".join([" ccc" for _ in classifiers])
    table_tex += f"\\begin{{tabular}}{{{col_format}}}\n"
    table_tex += "\\toprule\n"
    
    # Header Row 1: Classifier Names (multicolumns)
    table_tex += "\\textbf{Dataset}"
    for c in classifiers:
        table_tex += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{escape_tex(c)}}}}}"
    table_tex += " \\\\\n"
    
    # Header Row 2: Metrics
    table_tex += " "
    for _ in classifiers:
        table_tex += " & Acc & F1 & ms"
    table_tex += " \\\\\n"
    
    # Midrules for each classifier
    table_tex += "\\cmidrule(lr){1-1} "
    start_col = 2
    for _ in classifiers:
        table_tex += f"\\cmidrule(lr){{{start_col}-{start_col+2}}} "
        start_col += 3
    table_tex += "\n"
    
    # Data Rows
    for ds in datasets:
        row_tex = f"{escape_tex(ds)}"
        
        # We need to find the max accuracy to bold it
        accs = []
        f1s = []
        for c in classifiers:
            row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
            if len(row) > 0:
                accs.append(row["accuracy"].values[0])
                f1s.append(row["f1"].values[0])
            else:
                accs.append(-1)
                f1s.append(-1)
                
        max_acc = max(accs) if accs else -1
        max_f1 = max(f1s) if f1s else -1
        
        idx = 0
        for c in classifiers:
            row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
            if len(row) > 0:
                acc = row["accuracy"].values[0]
                f1 = row["f1"].values[0]
                ms = int(row["train_ms"].values[0])
                
                acc_str = f"\\textbf{{{acc:.4f}}}" if acc == max_acc and acc != -1 else f"{acc:.4f}"
                f1_str = f"\\textbf{{{f1:.4f}}}" if f1 == max_f1 and f1 != -1 else f"{f1:.4f}"
                
                row_tex += f" & {acc_str} & {f1_str} & {ms}"
            else:
                row_tex += " & -- & -- & --"
            idx += 1
            
        row_tex += " \\\\\n"
        table_tex += row_tex
        
    table_tex += "\\bottomrule\n"
    table_tex += "\\end{tabular}\n}\n\\end{table*}\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table_tex)
    print(f"Reporte LaTeX unificado guardado en {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generar Reporte LaTeX")
    parser.add_argument("--csv", default="benchmark_results.csv")
    parser.add_argument("--out", default="report.tex")
    args = parser.parse_args()
    generate_report(args.csv, args.out)
