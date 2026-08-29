#!/usr/bin/env python3
"""
scripts/plot_mgda.py — Visualización de la dinámica de entrenamiento MGDA.

Genera tres paneles:
  1. Trayectoria Pareto (Error vs Complejidad):  camino del gradiente
     desde la inicialización hasta el punto de convergencia Pareto.
  2. Coeficiente α(t) por iteración: muestra cuándo el MGDA "balancea"
     ambos gradientes y cuándo predomina uno u otro.
  3. Convergencia ||d||(t): escala log, detecta el instante Pareto
     cuando la norma de la dirección combinada colapsa.

Requiere: pip install matplotlib pandas
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Estética ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#9B5DE5']

def load_trajectory(path="mgda_trajectory.csv"):
    if not os.path.exists(path):
        print(f"[Error] Archivo no encontrado: {path}")
        print("Ejecuta primero: ./mgda ../data/moons_2d.csv")
        sys.exit(1)
    return pd.read_csv(path)

def main():
    df = load_trajectory()
    classes = sorted(df["clase"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("AM-MGDA — Dinámicas de Descenso Multi-Objetivo (MGDA)",
                 fontsize=13, fontweight='bold')
    ax1, ax2, ax3 = axes

    # ─── Panel 1: Trayectoria en el espacio Pareto ───────────
    ax1.set_title("Trayectoria Pareto\n(○ inicio  ★ convergencia)", fontsize=10)
    for i, cls in enumerate(classes):
        sub = df[df["clase"] == cls].reset_index(drop=True)
        c   = PALETTE[i % len(PALETTE)]
        ax1.plot(sub["error"], sub["complexity"],
                 '-', color=c, alpha=0.75, linewidth=1.8, label=f"Clase {cls}")
        # Inicio
        ax1.scatter(sub["error"].iloc[0], sub["complexity"].iloc[0],
                    marker='o', s=80, color=c, zorder=5, edgecolors='k', linewidths=0.7)
        # Punto Pareto final
        ax1.scatter(sub["error"].iloc[-1], sub["complexity"].iloc[-1],
                    marker='*', s=200, color=c, zorder=6, edgecolors='k', linewidths=0.7)

    ax1.set_xlabel("Error empírico (O₁)", fontsize=11)
    ax1.set_ylabel("Complejidad topológica (O₂)", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ─── Panel 2: Alpha (mezcla MGDA) por iteración ──────────
    ax2.set_title("Coeficiente MGDA α(t)\n(α→cte = tensión Pareto activa)", fontsize=10)
    for i, cls in enumerate(classes):
        sub = df[df["clase"] == cls]
        ax2.plot(sub["iter"], sub["alpha"],
                 '-', color=PALETTE[i % len(PALETTE)],
                 label=f"Clase {cls}", linewidth=1.8)

    ax2.axhline(1.0, color='limegreen',  linestyle='--', alpha=0.6, linewidth=1.2,
                label='α=1 (solo Error)')
    ax2.axhline(0.0, color='deepskyblue', linestyle='--', alpha=0.6, linewidth=1.2,
                label='α=0 (solo Complejidad)')
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.0,
                label='α=0.5 (equilibrio)')
    ax2.set_xlabel("Iteración", fontsize=11)
    ax2.set_ylabel("α (mezcla MGDA)", fontsize=11)
    ax2.set_ylim(-0.05, 1.10)
    ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.3)

    # ─── Panel 3: Norma del gradiente (convergencia) ─────────
    ax3.set_title("Convergencia MGDA\n(||d|| → 0 = punto Pareto detectado)", fontsize=10)
    for i, cls in enumerate(classes):
        sub = df[df["clase"] == cls]
        gn  = sub["grad_norm"].clip(lower=1e-12)  # evitar log(0)
        ax3.semilogy(sub["iter"], gn,
                     '-', color=PALETTE[i % len(PALETTE)],
                     label=f"Clase {cls}", linewidth=1.8)

    ax3.set_xlabel("Iteración", fontsize=11)
    ax3.set_ylabel("||d|| (norma dirección combinada)", fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()
    out = "mgda_plot.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"✓ Guardado: {out}")

if __name__ == "__main__":
    main()
