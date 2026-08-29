#!/usr/bin/env python3
"""
visualize.py — Visualización 2D del Manifold Classifier.

Genera:
  1. Fronteras de decisión en el espacio 2D
  2. Frentes de Pareto (accuracy vs complexity)
  3. Comparación de accuracy entre clasificadores (solo si hay benchmark_results.csv)

Requiere: pip install matplotlib numpy pandas scikit-learn
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── Configuración visual ──────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = ['#228B22', '#00C8FF', '#4C72B0', '#DD8452', '#8172B2']

def load_csv(path):
    """Carga CSV: retorna (X, y) donde X = features, y = labels."""
    import pandas as pd
    df = pd.read_csv(path)
    y = df.iloc[:, -1].values.astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    # Normalizar a [0,1]
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    return X, y

def fourier_manifold_boundary(center, coefs, n_harmonics, dim, resolution=300):
    """
    Genera puntos de la curva de Fourier paramétrica.
    center: array (dim,)
    coefs:  array (2*N*dim,)  layout: [a_{0,k}, b_{0,k}, ..., a_{dim-1,k}, ...]
    """
    t_vals = np.linspace(0, 2*np.pi, resolution, endpoint=True)
    curve = np.zeros((resolution, dim))
    for j, t in enumerate(t_vals):
        p = center.copy()
        for k in range(1, n_harmonics + 1):
            for i in range(dim):
                idx = (k-1)*2*dim + 2*i
                a = coefs[idx]
                b = coefs[idx + 1]
                p[i] += a * np.cos(k*t) + b * np.sin(k*t)
        curve[j] = p
    return curve

def plot_2d_boundaries(dataset_path="data/moons_2d.csv"):
    """Visualiza el dataset 2D y las fronteras de Fourier."""
    X, y = load_csv(dataset_path)
    classes = np.unique(y)
    n_cls = len(classes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title("FM-NSGA-II — Fronteras de decisión", fontsize=18, fontweight="bold")

    # Puntos del dataset
    for k, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[k % len(PALETTE)],
                   alpha=0.7, s=25, label=f"Clase {cls}", zorder=3)

    # Grilla de clasificación o curva paramétrica
    # (Bordes removidos a petición)

    ax.set_xlabel("Feature 1 (normalizado)", fontsize=16)
    ax.set_ylabel("Feature 2 (normalizado)", fontsize=16)
    ax.legend(loc="upper right", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("build/plot_dataset.png", bbox_inches='tight')
    print("Guardado: build/plot_dataset.png")

def plot_benchmark_results(csv_path="benchmark_results.csv"):
    """Gráfico de barras comparando accuracy en todos los datasets."""
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"No encontrado: {csv_path}  (ejecutar ./benchmark primero)")
        return

    df = pd.read_csv(csv_path)
    datasets = df["dataset"].unique()
    classifiers = df["classifier"].unique()

    x = np.arange(len(datasets))
    width = 0.8 / len(classifiers)
    offset = np.linspace(-0.4, 0.4, len(classifiers), endpoint=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Comparación de Accuracy — FM-NSGA-II vs Baselines", fontsize=13)

    bars = []
    for ki, clf in enumerate(classifiers):
        accs = []
        for ds in datasets:
            row = df[(df["dataset"] == ds) & (df["classifier"] == clf)]
            accs.append(row["accuracy"].values[0] if len(row) else 0.0)
        CUSTOM_PALETTE = ["#4A90E2", "#00C8FF", "#F5A623", "#D0021B", "#00E5FF", "#50E3C2", "#8B572A"]
        if "Fourier" in clf:
            color = "limegreen"
            alpha_val = 1.0
        elif "Angular" in clf:
            color = "#00C8FF"
            alpha_val = 1.0
        elif "GAM" in clf:
            color = "#00E5FF"
            alpha_val = 1.0
        else:
            color = CUSTOM_PALETTE[ki % len(CUSTOM_PALETTE)]
            alpha_val = 0.45
        b = ax.bar(x + offset[ki], accs, width, label=clf[:20],
                   color=color, alpha=alpha_val, edgecolor='white')
        bars.append(b)

    ax.set_xticks(x)
    ax.set_xticklabels([d[:18] for d in datasets], rotation=15, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_benchmark.png", bbox_inches='tight')
    plt.close()
    print("Guardado: plot_benchmark.png")

def plot_time_results(csv_path="benchmark_results.csv"):
    """Gráfico de barras comparando el tiempo de entrenamiento en escala logarítmica."""
    import pandas as pd
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    datasets = df["dataset"].unique()
    classifiers = df["classifier"].unique()

    x = np.arange(len(datasets))
    width = 0.8 / len(classifiers)
    offset = np.linspace(-0.4, 0.4, len(classifiers), endpoint=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Comparación de Tiempo de Entrenamiento (ms) — Escala Logarítmica", fontsize=13)

    for ki, clf in enumerate(classifiers):
        times = []
        for ds in datasets:
            row = df[(df["dataset"] == ds) & (df["classifier"] == clf)]
            t = row["train_ms"].values[0] if len(row) else 1.0
            # Evitar log(0) sumando 1ms
            times.append(max(t, 1.0))
        CUSTOM_PALETTE = ["#4A90E2", "#00C8FF", "#F5A623", "#D0021B", "#00E5FF", "#50E3C2", "#8B572A"]
        if "Fourier" in clf:
            color = "limegreen"
            alpha_val = 1.0
        elif "Angular" in clf:
            color = "#00C8FF"
            alpha_val = 1.0
        elif "GAM" in clf:
            color = "#00E5FF"
            alpha_val = 1.0
        else:
            color = CUSTOM_PALETTE[ki % len(CUSTOM_PALETTE)]
            alpha_val = 0.45
        ax.bar(x + offset[ki], times, width, label=clf[:20],
               color=color, alpha=alpha_val, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([d[:18] for d in datasets], rotation=15, ha='right')
    ax.set_ylabel("Tiempo de entrenamiento (ms)")
    ax.set_yscale('log')
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis='y', alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig("plot_time.png", bbox_inches='tight')
    plt.close()
    print("Guardado: plot_time.png")


def plot_pareto_front_example():
    """Muestra un frente de Pareto de ejemplo (error vs complejidad)."""
    # Datos sintéticos de ejemplo
    np.random.seed(7)
    n_pts = 30
    # Frente de Pareto real (monótonamente decreciente)
    complexity = np.sort(np.random.uniform(0.05, 0.95, n_pts))
    # Ordenar error descendentemente para que al aumentar complejidad, el error caiga
    error = np.sort(0.5 - 0.45 * complexity + np.random.normal(0, 0.05, n_pts))[::-1]
    error = np.clip(error, 0.01, 0.99)

    # Puntos dominados: se generan sumando ruido positivo a puntos del frente
    idx = np.random.choice(n_pts, 50)
    cx_dom = np.clip(complexity[idx] + np.random.uniform(0.02, 0.2, 50), 0, 1)
    er_dom = np.clip(error[idx] + np.random.uniform(0.02, 0.2, 50), 0, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Frente de Pareto: Error vs Complejidad del Manifold", fontsize=12)
    ax.scatter(cx_dom, er_dom, c='#ADB5BD', alpha=0.5, s=20, label="Dominados", zorder=2)
    ax.plot(complexity, error, 'o-', color='#E63946', ms=7, lw=2,
            label="Frente de Pareto", zorder=3)

    # Anotar puntos especiales
    ax.annotate("Máx. precisión\n(más complejo)", xy=(complexity[-1], error[-1]),
                xytext=(0.65, 0.35), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate("Mín. complejidad\n(más simple)", xy=(complexity[0], error[0]),
                xytext=(0.02, 0.55), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel("Complejidad del Manifold (normalizada)")
    ax.set_ylabel("Tasa de Error de Clasificación")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_pareto.png", bbox_inches='tight')
    plt.close()
    print("Guardado: plot_pareto.png")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando visualizaciones...")

    # 1. Dataset 2D
    for ds in ["../data/chessboard_2d.csv", "../data/blobs_2d.csv", "../data/moons_2d.csv", "../data/circles_2d.csv", "../data/spirals_2d.csv", "data/chessboard_2d.csv", "data/blobs_2d.csv", "data/moons_2d.csv", "data/circles_2d.csv", "data/spirals_2d.csv"]:
        if os.path.exists(ds):
            plot_2d_boundaries(ds)
            plt.close('all')
            break

    # 2. Benchmark (si existe)
    plot_benchmark_results()
    plot_time_results()

    # 3. Frente de Pareto (ejemplo)
    plot_pareto_front_example()

    print("\n✓ Figuras generadas.")
    if "--no-show" not in sys.argv:
        plt.show()
