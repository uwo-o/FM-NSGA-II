import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "build/pareto_front.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} no encontrado.")
    exit(1)

df = pd.read_csv(csv_path)
classes = df["clase"].unique()

plt.figure(figsize=(8, 6))
plt.title("Frente de Pareto (NSGA-II)", fontsize=14, fontweight="bold")
plt.xlabel("Objetivo 1: Error Empírico (Tasa de fallos)", fontsize=12)
plt.ylabel("Objetivo 2: Complejidad Topológica (Normalizada)", fontsize=12)

for i, cls in enumerate(classes):
    subset = df[df["clase"] == cls]
    dom = subset[subset["rank"] > 1]
    nondom = subset[subset["rank"] == 1]
    
    # Soluciones dominadas (fondo/grises)
    plt.scatter(dom["error"], dom["complejidad"], 
                color="lightgray", alpha=0.5, s=30, label="Dominadas" if i==0 else "")
                
    # Soluciones no dominadas (verde lima)
    plt.scatter(nondom["error"], nondom["complejidad"], 
                label=f"No dominadas (Clase {cls})", color="limegreen", alpha=0.9, edgecolors='k', s=60)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="One-vs-Rest")
plt.tight_layout()
plt.savefig("build/plot_pareto.png", dpi=300, bbox_inches='tight')
print("Guardado en build/plot_pareto.png")
