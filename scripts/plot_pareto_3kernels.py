import pandas as pd
import matplotlib.pyplot as plt
import os

files = {
    "NSGA-Fourier": "build/pareto_front_NSGA-Fourier.csv",
    "NSGA-Angular": "build/pareto_front_NSGA-Angular.csv",
    "NSGA-GAM":     "build/pareto_front_NSGA-GAM.csv",
    "NSGA-SHM":     "build/pareto_front_NSGA-SHM.csv"
}

colors = {
    "NSGA-Fourier": "#7FFF00",   # Verde lima
    "NSGA-Angular": "#9B30FF",   # Morado
    "NSGA-GAM":     "#00C8FF",   # Celeste eléctrico
    "NSGA-SHM":     "#FF3366",   # Rosa eléctrico (SHM)
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Frente de Pareto Unificado (4 Kernels NSGA-II)", fontsize=18, fontweight="bold")
ax.set_xlabel("Objetivo 1: Error Empírico (Tasa de fallos)", fontsize=16)
ax.set_ylabel("Objetivo 2: Complejidad Topológica", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

for name, path in files.items():
    if not os.path.exists(path):
        print(f"Error: {path} no encontrado.")
        continue
    df = pd.read_csv(path)
    nondom = df[df["rank"] == 1].copy()
    # Clamp values to avoid log(0): add small epsilon
    import numpy as np
    eps = 1e-4
    nondom["error"] = nondom["error"].clip(lower=eps)
    nondom["complejidad"] = nondom["complejidad"].clip(lower=eps)
    nondom = nondom.sort_values("error")

    ax.scatter(nondom["error"], nondom["complejidad"],
               label=name, color=colors[name],
               marker='o', alpha=0.95, edgecolors='white', s=90, zorder=3)

ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=14)
fig.tight_layout()
fig.savefig("build/plot_pareto_3kernels.png", dpi=300, bbox_inches='tight')
print("Guardado en build/plot_pareto_3kernels.png")
