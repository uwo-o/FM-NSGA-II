import pandas as pd

def escape_tex(text):
    return text.replace('_', r'\_').replace('%', r'\%').replace('γ', r'$\gamma$')

df = pd.read_csv("../build/variance_results.csv")
datasets = df["dataset"].unique()
classifiers = df["classifier"].unique()

# --- TABLA DE ACCURACY ---
table_acc = "\\begin{table*}[t]\n\\centering\n"
table_acc += "\\caption{Precisión global (Accuracy) en la batería de pruebas calculada a lo largo de 20 ejecuciones independientes. Se reporta la media y la desviación estándar ($\pm \sigma$) para exponer la naturaleza estocástica del FM-NSGA-II. Se evidencia cómo los modelos lineales colapsan en distribuciones complejas (Circles, Two Moons), mientras que el FM-NSGA-II logra mantener un rendimiento top-tier consistente, salvo frente al ruido extremo de la maldición de la dimensionalidad (Adversarial 52D) donde los Decision Trees toman ventaja por su Feature Selection inherente. El mejor desempeño medio por dataset se resalta en negrita.}\n"
table_acc += "\\vspace{0.2cm}\n"
table_acc += "\\resizebox{\\textwidth}{!}{\n"
table_acc += "\\begin{tabular}{l" + "c" * len(classifiers) + "}\n"
table_acc += "\\toprule\n"
table_acc += "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]) + " \\\\\n"
table_acc += "\\midrule\n"
for ds in datasets:
    row_tex = f"\\textbf{{{escape_tex(ds)}}}"
    accs = []
    stds = []
    for c in classifiers:
        row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
        if len(row) > 0:
            accs.append(row["accuracy_mean"].values[0])
            stds.append(row["accuracy_std"].values[0])
        else:
            accs.append(-1)
            stds.append(0)
            
    max_acc = max(accs)
    for val, std in zip(accs, stds):
        if val == -1:
            row_tex += " & --"
        else:
            fmt_str = f"{val:.4f} $\\pm$ {std:.4f}" if std > 0.0001 else f"{val:.4f}"
            if val == max_acc:
                row_tex += f" & \\textbf{{{fmt_str}}}"
            else:
                row_tex += f" & {fmt_str}"
    row_tex += " \\\\\n"
    table_acc += row_tex
table_acc += "\\bottomrule\n"
table_acc += "\\end{tabular}\n}\n\\label{tab:accuracy}\n\\end{table*}\n"

# --- TABLA DE TIEMPOS ---
table_time = "\\begin{table*}[t]\n\\centering\n"
table_time += "\\caption{Costo computacional de convergencia y tiempos de entrenamiento empírico (expresados en milisegundos $\pm \sigma$ tras 20 iteraciones). Al tratarse de un algoritmo heurístico poblacional (NSGA-II) que debe evaluar distancias proyectadas ortogonalmente a un colector paramétrico complejo, el FM-NSGA-II expone un orden de complejidad temporal manifiestamente superior frente a los clasificadores deterministas clásicos, logrando viabilidad únicamente a través de paralelismo asíncrono y mini-batching evolutivo.}\n"
table_time += "\\vspace{0.2cm}\n"
table_time += "\\resizebox{\\textwidth}{!}{\n"
table_time += "\\begin{tabular}{l" + "r" * len(classifiers) + "}\n"
table_time += "\\toprule\n"
table_time += "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{escape_tex(c)}}}" for c in classifiers]) + " \\\\\n"
table_time += "\\midrule\n"
for ds in datasets:
    row_tex = f"\\textbf{{{escape_tex(ds)}}}"
    for c in classifiers:
        row = df[(df["dataset"] == ds) & (df["classifier"] == c)]
        if len(row) > 0:
            mean_ms = row["train_ms_mean"].values[0]
            std_ms = row["train_ms_std"].values[0]
            fmt_str = f"{int(mean_ms)} $\\pm$ {int(std_ms)}" if std_ms >= 1 else f"{int(mean_ms)}"
            row_tex += f" & {fmt_str}"
        else:
            row_tex += " & --"
    row_tex += " \\\\\n"
    table_time += row_tex
table_time += "\\bottomrule\n"
table_time += "\\end{tabular}\n}\n\\label{tab:time}\n\\end{table*}\n"

with open("tables.tex", "w") as f:
    f.write(table_acc + "\n" + table_time)
print("tables.tex updated with variance.")
