import pandas as pd
import os

def escape_tex(text):
    return (text.replace('_', r'\_')
                .replace('%', r'\%')
                .replace('γ', r'$\gamma$'))

# ─── Cargar datos ──────────────────────────────────────────────
# Preferir variance_results.csv (20 corridas) si existe,
# si no usar benchmark_results.csv (corrida única)
variance_path = "../build/variance_results.csv"
single_path   = "../build/mgda_bench_results.csv"

use_variance = os.path.exists(variance_path)

if use_variance:
    df_raw = pd.read_csv(variance_path)
    # Columnas esperadas: dataset, classifier, accuracy_mean, accuracy_std,
    #                     f1_mean, f1_std, train_ms_mean, train_ms_std
    df = df_raw
    print("Usando variance_results.csv (20 corridas)")
else:
    df_raw = pd.read_csv(single_path)
    # Columnas: dataset, classifier, accuracy, f1, train_ms
    # Simular columnas de varianza con std=0
    df = df_raw.copy()
    df["accuracy_mean"] = df["accuracy"]
    df["accuracy_std"]  = 0.0
    df["f1_mean"]       = df["f1"]
    df["f1_std"]        = 0.0
    df["train_ms_mean"] = df["train_ms"]
    df["train_ms_std"]  = 0.0
    print(f"Usando {single_path} (1 corrida)")

datasets    = df["dataset"].unique()
classifiers = df["classifier"].unique()

# ─── Orden preferido de clasificadores ────────────────────────
preferred_order = [
    "NSGA-Fourier",
    "NSGA-Angular",
    "NSGA-GAM",
    "MGDA-Angular",
    "MGDA-GAM",
    "k-NN (k=3)",
    "k-NN (k=7)",
    "Gaussian Naive Bayes",
    "SVM-RBF (C=1 γ=1.00)",
    "SVM-Linear (C=1)",
    "Decision Tree (depth=8)",
]
# Ordenar respetando el orden preferido, luego los que no estén en él
classifiers = sorted(classifiers, key=lambda c: preferred_order.index(c)
                     if c in preferred_order else 999)

note = "(una corrida)" if not use_variance else "(media $\\pm\\sigma$, 20 corridas)"

# ─── TABLA DE ACCURACY ───────────────────────────────────────
caption_acc = (
    f"Precisión global (Accuracy) {note}. Se comparan los tres kernels topológicos "
    "evolucionados (FM, AM, GAM) contra baselines clásicos en 9 datasets. "
    "El mejor valor por dataset se resalta en negrita. "
    "GAM-NSGA-II domina en datos tabulares de alta dimensión "
    "(Breast Cancer 30D: 97.35\\%), mientras FM-NSGA-II mantiene "
    "la ventaja en geometrías no-aditivas (Two Moons, Wine). "
    "AM-NSGA-II ofrece la mayor velocidad a costa de precisión en alta dimensión."
)

table_acc  = "\\begin{table*}[t]\n\\centering\n"
table_acc += f"\\caption{{{caption_acc}}}\n"
table_acc += "\\vspace{0.2cm}\n"
table_acc += "\\resizebox{\\textwidth}{!}{\n"
table_acc += "\\begin{tabular}{l" + "c" * len(classifiers) + "}\n"
table_acc += "\\toprule\n"
table_acc += "\\textbf{Dataset} & " + " & ".join(
    [f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]
) + " \\\\\n"
table_acc += "\\midrule\n"

for ds in datasets:
    row_tex = f"\\textbf{{{escape_tex(ds)}}}"
    accs, stds = [], []
    for c in classifiers:
        row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
        if len(row) > 0:
            accs.append(row["accuracy_mean"].values[0])
            stds.append(row["accuracy_std"].values[0])
        else:
            accs.append(-1)
            stds.append(0)

    max_acc = max(a for a in accs if a >= 0)
    for val, std in zip(accs, stds):
        if val < 0:
            row_tex += " & --"
        else:
            fmt = f"{val:.4f} $\\pm$ {std:.4f}" if std > 0.0001 else f"{val:.4f}"
            row_tex += f" & \\textbf{{{fmt}}}" if val == max_acc else f" & {fmt}"
    row_tex += " \\\\\n"
    table_acc += row_tex

table_acc += "\\bottomrule\n"
table_acc += "\\end{tabular}\n}\n\\label{tab:accuracy}\n\\end{table*}\n"

# ─── TABLA DE F1 ─────────────────────────────────────────────
caption_f1 = (
    f"Macro F1-Score {note}. Complementa la Tabla~\\ref{{tab:accuracy}} "
    "con la métrica F1 para exponer desequilibrios en precisión/recall. "
    "GAM-NSGA-II logra F1=0.9682 en Breast Cancer 30D, superando a FM (0.9172) y AM (0.9257)."
)

table_f1  = "\\begin{table*}[t]\n\\centering\n"
table_f1 += f"\\caption{{{caption_f1}}}\n"
table_f1 += "\\vspace{0.2cm}\n"
table_f1 += "\\resizebox{\\textwidth}{!}{\n"
table_f1 += "\\begin{tabular}{l" + "c" * len(classifiers) + "}\n"
table_f1 += "\\toprule\n"
table_f1 += "\\textbf{Dataset} & " + " & ".join(
    [f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]
) + " \\\\\n"
table_f1 += "\\midrule\n"

for ds in datasets:
    row_tex = f"\\textbf{{{escape_tex(ds)}}}"
    f1s, stds = [], []
    for c in classifiers:
        row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
        if len(row) > 0:
            f1s.append(row["f1_mean"].values[0])
            stds.append(row["f1_std"].values[0])
        else:
            f1s.append(-1)
            stds.append(0)

    max_f1 = max(v for v in f1s if v >= 0)
    for val, std in zip(f1s, stds):
        if val < 0:
            row_tex += " & --"
        else:
            fmt = f"{val:.4f} $\\pm$ {std:.4f}" if std > 0.0001 else f"{val:.4f}"
            row_tex += f" & \\textbf{{{fmt}}}" if val == max_f1 else f" & {fmt}"
    row_tex += " \\\\\n"
    table_f1 += row_tex

table_f1 += "\\bottomrule\n"
table_f1 += "\\end{tabular}\n}\n\\label{tab:f1}\n\\end{table*}\n"

# ─── TABLA DE TIEMPOS ────────────────────────────────────────
caption_time = (
    f"Costo computacional de entrenamiento {note} (ms). "
    "AM-NSGA-II es el kernel más rápido ($\\sim$70\\,ms en 2D), "
    "FM-NSGA-II es el más costoso en alta dimensión (>57\\,000\\,ms en 30D), "
    "GAM-NSGA-II ofrece un balance intermedio ($\\sim$2\\,000\\,ms en 30D) "
    "con mayor precisión que FM en datos tabulares."
)

table_time  = "\\begin{table*}[t]\n\\centering\n"
table_time += f"\\caption{{{caption_time}}}\n"
table_time += "\\vspace{0.2cm}\n"
table_time += "\\resizebox{\\textwidth}{!}{\n"
table_time += "\\begin{tabular}{l" + "r" * len(classifiers) + "}\n"
table_time += "\\toprule\n"
table_time += "\\textbf{Dataset} & " + " & ".join(
    [f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]
) + " \\\\\n"
table_time += "\\midrule\n"

for ds in datasets:
    row_tex = f"\\textbf{{{escape_tex(ds)}}}"
    for c in classifiers:
        row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
        if len(row) > 0:
            mean_ms = row["train_ms_mean"].values[0]
            std_ms  = row["train_ms_std"].values[0]
            fmt = (f"{int(mean_ms)} $\\pm$ {int(std_ms)}"
                   if std_ms >= 1 else f"{int(mean_ms)}")
            row_tex += f" & {fmt}"
        else:
            row_tex += " & --"
    row_tex += " \\\\\n"
    table_time += row_tex

table_time += "\\bottomrule\n"
table_time += "\\end{tabular}\n}\n\\label{tab:time}\n\\end{table*}\n"

# ─── Guardar ─────────────────────────────────────────────────
with open("tables.tex", "w") as f:
    f.write(table_acc + "\n" + table_f1 + "\n" + table_time)

print(f"tables.tex actualizado — {len(datasets)} datasets, {len(classifiers)} clasificadores.")
