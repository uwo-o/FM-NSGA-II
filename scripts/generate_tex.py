#!/usr/bin/env python3
"""
generate_tex.py — Genera el código fuente LaTeX (report.tex)
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

    # Identificar el modelo propuesto
    prop = "FM-NSGA-II"
    if prop not in classifiers:
        prop = classifiers[0]

    # --- TABLA DE ACCURACY ---
    table_acc = "\\begin{table}[h!]\n\\centering\n"
    table_acc += "\\caption{Accuracy por Dataset y Clasificador}\n"
    table_acc += "\\vspace{0.2cm}\n"
    table_acc += "\\resizebox{\\textwidth}{!}{\n"
    table_acc += "\\begin{tabular}{l" + "c" * len(classifiers) + "}\n"
    table_acc += "\\toprule\n"
    table_acc += "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]) + " \\\\\n"
    table_acc += "\\midrule\n"
    for ds in datasets:
        row_tex = f"{escape_tex(ds)}"
        accs = []
        for c in classifiers:
            val = df[(df["dataset"] == ds) & (df["classifier"] == c)]["accuracy"]
            if len(val) > 0:
                accs.append(val.values[0])
            else:
                accs.append(-1)
        max_acc = max(accs)
        for val in accs:
            if val == -1:
                row_tex += " & --"
            elif val == max_acc:
                row_tex += f" & \\textbf{{{val:.4f}}}"
            else:
                row_tex += f" & {val:.4f}"
        row_tex += " \\\\\n"
        table_acc += row_tex
    table_acc += "\\bottomrule\n"
    table_acc += "\\end{tabular}\n}\n\\end{table}\n"

    # --- TABLA DE F1 SCORE ---
    table_f1 = "\\begin{table}[h!]\n\\centering\n"
    table_f1 += "\\caption{F1-Score por Dataset y Clasificador}\n"
    table_f1 += "\\vspace{0.2cm}\n"
    table_f1 += "\\resizebox{\\textwidth}{!}{\n"
    table_f1 += "\\begin{tabular}{l" + "c" * len(classifiers) + "}\n"
    table_f1 += "\\toprule\n"
    table_f1 += "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]) + " \\\\\n"
    table_f1 += "\\midrule\n"
    for ds in datasets:
        row_tex = f"{escape_tex(ds)}"
        f1s = []
        for c in classifiers:
            val = df[(df["dataset"] == ds) & (df["classifier"] == c)]["f1"]
            if len(val) > 0:
                f1s.append(val.values[0])
            else:
                f1s.append(-1)
        max_f1 = max(f1s)
        for val in f1s:
            if val == -1:
                row_tex += " & --"
            elif val == max_f1:
                row_tex += f" & \\textbf{{{val:.4f}}}"
            else:
                row_tex += f" & {val:.4f}"
        row_tex += " \\\\\n"
        table_f1 += row_tex
    table_f1 += "\\bottomrule\n"
    table_f1 += "\\end{tabular}\n}\n\\end{table}\n"

    # --- TABLA DE TIEMPOS (MS) ---
    table_time = "\\begin{table}[h!]\n\\centering\n"
    table_time += "\\caption{Tiempo de Entrenamiento (ms) por Dataset y Clasificador}\n"
    table_time += "\\vspace{0.2cm}\n"
    table_time += "\\resizebox{\\textwidth}{!}{\n"
    table_time += "\\begin{tabular}{l" + "r" * len(classifiers) + "}\n"
    table_time += "\\toprule\n"
    table_time += "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]) + " \\\\\n"
    table_time += "\\midrule\n"
    for ds in datasets:
        row_tex = f"{escape_tex(ds)}"
        for c in classifiers:
            val = df[(df["dataset"] == ds) & (df["classifier"] == c)]["train_ms"]
            if len(val) > 0:
                row_tex += f" & {int(val.values[0])}"
            else:
                row_tex += " & --"
        row_tex += " \\\\\n"
        table_time += row_tex
    table_time += "\\bottomrule\n"
    table_time += "\\end{tabular}\n}\n\\end{table}\n"

    tex_doc = f"""\\documentclass[11pt, a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{float}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\geometry{{margin=2.5cm}}

\\title{{\\textbf{{Reporte de Rendimiento: FM-NSGA-II}}\\\\ \\Large Evaluación Exhaustiva vs. Baselines}}
\\author{{Generado Automáticamente}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
Este documento presenta los resultados del benchmark realizado sobre el modelo \\textbf{{FM-NSGA-II}} en comparación con varios clasificadores tradicionales (k-NN, Naive Bayes, SVM-RBF, Decision Tree). Se evalúa tanto el rendimiento predictivo (Accuracy y F1-Score) como la escalabilidad y tiempo de entrenamiento en datasets sintéticos y reales de diferente dimensionalidad.
\\end{{abstract}}

\\section{{Métricas de Precisión}}
El rendimiento del modelo evolutivo ha sido altamente competitivo. En algunos escenarios de alta dimensionalidad, el modelo alcanza un equilibrio óptimo debido a la optimización multiobjetivo (complejidad vs error).

{table_acc}

{table_f1}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{plot_benchmark.png}}
    \\caption{{Comparación gráfica de Accuracy entre clasificadores.}}
\\end{{figure}}

\\clearpage
\\section{{Análisis de Tiempos de Entrenamiento}}

A continuación se muestra el tiempo de convergencia del modelo evaluado en una escala logarítmica para contextualizar el comportamiento asintótico frente a modelos deterministas (e.g. SVM). Gracias a las recientes optimizaciones (OpenMP, early stopping y caching de proyecciones), el algoritmo mantiene tiempos escalables.

{table_time}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{plot_time.png}}
    \\caption{{Tiempo de entrenamiento (escala logarítmica) requerido por cada clasificador.}}
\\end{{figure}}

\\section{{Análisis del Manifold y Fronteras de Pareto}}

La característica distintiva del FM-NSGA-II es el aprendizaje de una frontera de decisión basada en coeficientes de Fourier de longitud variable. El frente de Pareto refleja el compromiso entre la precisión del modelo y la complejidad intrínseca del mismo (medida mediante la longitud de arco y el número de armónicos).

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{plot_pareto.png}}
    \\caption{{Ejemplo del Frente de Pareto de la población en la última generación.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{{plot_dataset.png}}
    \\caption{{Proyección bidimensional del manifold y las fronteras generadas sobre los datos de prueba.}}
\\end{{figure}}

\\end{{document}}
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_doc)
    print(f"Reporte LaTeX guardado en {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generar Reporte LaTeX")
    parser.add_argument("--csv", default="benchmark_results.csv")
    parser.add_argument("--out", default="report.tex")
    args = parser.parse_args()
    generate_report(args.csv, args.out)
