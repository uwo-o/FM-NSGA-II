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

def plot_importance():
    csv_path = "build/gam_weights.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return
        
    df_pivot = df.pivot(index="Feature", columns="Method", values="Weight").reset_index()
    df_pivot["mean_w"] = (df_pivot["MGDA"] + df_pivot["NSGA-II"]) / 2
    df_pivot = df_pivot.sort_values(by="mean_w", ascending=True).tail(15)

    features = df_pivot["Feature"].tolist()
    mgda_w = df_pivot["MGDA"].tolist()
    nsga_w = df_pivot["NSGA-II"].tolist()

    y = np.arange(len(features))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.barh(y - height/2, mgda_w, height, label='MGDA', color='#228B22')
    rects2 = ax.barh(y + height/2, nsga_w, height, label='NSGA-II', color='#00FF00')

    ax.set_xlabel('Weight ($w_i$)', fontsize=16)
    ax.set_ylabel('Feature', fontsize=16)
    ax.set_title('GAM Feature Importance (SHAP-style)\nTop 15 Features on BreastCancer', fontsize=18, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(title="Optimization Method", loc="lower right", fontsize=14, title_fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig("build/gam_importance.png", dpi=200, bbox_inches="tight")
    print("Saved build/gam_importance.png")

if __name__ == "__main__":
    plot_importance()
