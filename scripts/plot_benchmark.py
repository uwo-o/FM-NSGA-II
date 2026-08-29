#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color único y distintivo para CADA algoritmo (todos, no solo los nuestros)
COLORS = {
    "MGDA-Angular":             "#00C8FF",  # Celeste eléctrico
    "MGDA-GAM":                 "#00E000",  # Verde eléctrico
    "NSGA-Angular":             "#FF6B35",  # Naranja
    "NSGA-GAM":                 "#FFD700",  # Dorado
    "NSGA-Fourier":             "#FF1493",  # Magenta
    "k-NN (k=3)":               "#A855F7",  # Violeta
    "k-NN (k=7)":               "#C084FC",  # Lila
    "Gaussian Naive Bayes":     "#F97316",  # Naranja oscuro
    "SVM-RBF (C=1 γ=1.00)":    "#14B8A6",  # Teal
    "Decision Tree (depth=8)":  "#F43F5E",  # Rosa coral
}

TARGET_CLASSIFIERS = [
    "MGDA-Angular", "MGDA-GAM",
    "NSGA-Angular", "NSGA-GAM", "NSGA-Fourier",
    "k-NN (k=3)", "k-NN (k=7)", "Gaussian Naive Bayes",
    "SVM-RBF (C=1 γ=1.00)", "Decision Tree (depth=8)"
]

ALPHA_WINNER = 1.0
ALPHA_REST   = 0.30

def plot_benchmark():
    csv_path = "build/mgda_bench_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    target_datasets = df['dataset'].unique()
    df_sub = df[df["classifier"].isin(TARGET_CLASSIFIERS)].copy()

    x = np.arange(len(target_datasets))
    width = 0.08

    # ── 1. Accuracy ───────────────────────────────────────────────────────
    # Precalcular ganador por dataset (accuracy más alta)
    acc_matrix = {}
    for clf in TARGET_CLASSIFIERS:
        row = []
        for ds in target_datasets:
            val = df_sub[(df_sub["classifier"] == clf) & (df_sub["dataset"] == ds)]["accuracy"].values
            row.append(val[0] if len(val) > 0 else 0.0)
        acc_matrix[clf] = row

    # Ganadores por dataset = TODOS los clf con accuracy igual al máximo (empates incluidos)
    winners_per_ds = []
    tol = 1e-4  # tolerancia para empates
    for j in range(len(target_datasets)):
        best_val = max(acc_matrix[c][j] for c in TARGET_CLASSIFIERS)
        winners = {c for c in TARGET_CLASSIFIERS if acc_matrix[c][j] >= best_val - tol}
        winners_per_ds.append(winners)

    fig, ax = plt.subplots(figsize=(18, 6))

    for i, clf in enumerate(TARGET_CLASSIFIERS):
        color = COLORS.get(clf, "#888888")
        offset = (i - len(TARGET_CLASSIFIERS) / 2.0) * width + width / 2.0

        # Construir barras una por una para aplicar alpha variable
        for j, (ds, acc) in enumerate(zip(target_datasets, acc_matrix[clf])):
            alpha = ALPHA_WINNER if clf in winners_per_ds[j] else ALPHA_REST
            ax.bar(x[j] + offset, acc, width,
                   color=color, alpha=alpha, edgecolor='white', linewidth=0.6,
                   label=clf if j == 0 else "_nolegend_")

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy across All Datasets', fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(target_datasets, rotation=45, ha='right')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=5, fontsize=9)
    ax.set_ylim(0.0, 1.10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    plt.savefig("benchmark_acc.png", dpi=200, bbox_inches="tight")
    print("Saved benchmark_acc.png")

    # ── 2. Training Time (Log, ganador = menor tiempo) ─────────────────────
    time_matrix = {}
    for clf in TARGET_CLASSIFIERS:
        row = []
        for ds in target_datasets:
            val = df_sub[(df_sub["classifier"] == clf) & (df_sub["dataset"] == ds)]["train_ms"].values
            row.append(max(val[0], 1) if len(val) > 0 else 1.0)
        time_matrix[clf] = row

    # Ganadores de tiempo por dataset = los que tienen menor tiempo (con tolerancia)
    winners_time_per_ds = []
    for j in range(len(target_datasets)):
        best_t = min(time_matrix[c][j] for c in TARGET_CLASSIFIERS)
        winners = {c for c in TARGET_CLASSIFIERS if time_matrix[c][j] <= best_t * (1 + tol)}
        winners_time_per_ds.append(winners)

    fig, ax = plt.subplots(figsize=(18, 6))

    for i, clf in enumerate(TARGET_CLASSIFIERS):
        color = COLORS.get(clf, "#888888")
        offset = (i - len(TARGET_CLASSIFIERS) / 2.0) * width + width / 2.0

        for j, (ds, t) in enumerate(zip(target_datasets, time_matrix[clf])):
            alpha = ALPHA_WINNER if clf in winners_time_per_ds[j] else ALPHA_REST
            ax.bar(x[j] + offset, t, width,
                   color=color, alpha=alpha, edgecolor='white', linewidth=0.6,
                   label=clf if j == 0 else "_nolegend_")

    ax.set_ylabel('Training Time (ms) - Log Scale', fontsize=12)
    ax.set_title('Training Time across All Datasets', fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(target_datasets, rotation=45, ha='right')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=5, fontsize=9)
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')

    fig.tight_layout()
    plt.savefig("benchmark_time.png", dpi=200, bbox_inches="tight")
    print("Saved benchmark_time.png")

if __name__ == "__main__":
    plot_benchmark()
