import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 2.5))

# Hide axes
ax.axis('off')

# Parameters
n_genes_to_show = 10
cell_width = 1.0
cell_height = 0.6
start_x = 0
start_y = 0.5

# Colors
colors = ["#E63946", "#E63946"] + ["#E9C46A", "#F4A261", "#2A9D8F", "#264653"] * 2

labels = [
    r"$C_x$", r"$C_y$",
    r"$A_1^x$", r"$B_1^x$", r"$A_1^y$", r"$B_1^y$",
    r"$A_2^x$", r"$B_2^x$", r"$A_2^y$", r"$B_2^y$"
]

values = [
    "0.06", "0.63",
    "0.49", "0.46", "0.23", "-0.31",
    "0.44", "0.45", "0.00", "-0.31"
]

ax.text(start_x + (n_genes_to_show * cell_width)/2, start_y + cell_height + 0.3,
        "Estructura del Cromosoma (Individuo NSGA-II)",
        fontsize=14, fontweight='bold', ha='center')

for i in range(n_genes_to_show):
    # Draw cell
    rect = patches.Rectangle((start_x + i * cell_width, start_y), cell_width, cell_height, 
                             linewidth=2, edgecolor='black', facecolor=colors[i], alpha=0.3)
    ax.add_patch(rect)
    
    # Gene Label
    ax.text(start_x + i * cell_width + cell_width/2, start_y + cell_height/2 + 0.15,
            labels[i], fontsize=14, ha='center', va='center', fontweight='bold')
    
    # Gene Value
    ax.text(start_x + i * cell_width + cell_width/2, start_y + cell_height/2 - 0.15,
            values[i], fontsize=12, ha='center', va='center')

# Ellipsis for the rest of the harmonics
ax.text(start_x + n_genes_to_show * cell_width + 0.5, start_y + cell_height/2,
        r"$\dots$", fontsize=20, ha='center', va='center')

# Last harmonic
i = n_genes_to_show + 1
last_labels = [r"$A_H^x$", r"$B_H^x$", r"$A_H^y$", r"$B_H^y$"]
last_values = ["0.01", "-0.02", "0.00", "0.03"]

for j in range(4):
    idx = i + j
    rect = patches.Rectangle((start_x + idx * cell_width, start_y), cell_width, cell_height, 
                             linewidth=2, edgecolor='black', facecolor=colors[j], alpha=0.3)
    ax.add_patch(rect)
    
    ax.text(start_x + idx * cell_width + cell_width/2, start_y + cell_height/2 + 0.15,
            last_labels[j], fontsize=14, ha='center', va='center', fontweight='bold')
    
    ax.text(start_x + idx * cell_width + cell_width/2, start_y + cell_height/2 - 0.15,
            last_values[j], fontsize=12, ha='center', va='center')

# Annotations
ax.annotate('Centroide ($a_0$)', xy=(start_x + cell_width, start_y), xytext=(start_x + cell_width, start_y - 0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            fontsize=12, ha='center')

ax.annotate('Armónico 1', xy=(start_x + 4*cell_width, start_y), xytext=(start_x + 4*cell_width, start_y - 0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            fontsize=12, ha='center')

ax.annotate('Armónico 2', xy=(start_x + 8*cell_width, start_y), xytext=(start_x + 8*cell_width, start_y - 0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            fontsize=12, ha='center')

ax.annotate('Armónico H (max)', xy=(start_x + (i+2)*cell_width, start_y), xytext=(start_x + (i+2)*cell_width, start_y - 0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            fontsize=12, ha='center')

ax.set_xlim(start_x - 0.5, start_x + (n_genes_to_show + 5) * cell_width)
ax.set_ylim(start_y - 0.6, start_y + cell_height + 0.6)

plt.tight_layout()
plt.savefig("../build/plot_chromosome.png", dpi=300, bbox_inches='tight')
print("Guardado: plot_chromosome.png")
