import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(3, 1, figsize=(14, 8))
fig.patch.set_facecolor('#F8F9FA')

for ax in axes:
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

cell_w = 1.0
cell_h = 0.55
start_x = 0.0
start_y = 0.2

COLOR_CENTER  = "#E63946"   # rojo – centroide
COLOR_V       = "#457B9D"   # azul – vector direccional
COLOR_A       = "#2A9D8F"   # verde – coeficientes a_k
COLOR_WEIGHTS = "#6A4C93"   # violeta – pesos w

# ─── FILA 1: Centroide c y Vector Direccional v ───────────────────────────
ax1 = axes[0]
ax1.text(7.5, start_y + cell_h + 0.32,
         "Estructura del Cromosoma — Angular Manifold",
         fontsize=14, fontweight='bold', ha='center', color='#1D1D1D')

genes_row1 = [
    (r"$c_1$", "0.06", COLOR_CENTER), (r"$c_2$", "0.63", COLOR_CENTER),
    (r"$c_3$", "0.21", COLOR_CENTER), (r"$c_4$", "-0.1", COLOR_CENTER),
]
for i, (lbl, val, col) in enumerate(genes_row1):
    rect = patches.FancyBboxPatch((start_x + i * cell_w + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=col, alpha=0.35)
    ax1.add_patch(rect)
    ax1.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.65, lbl, fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
    ax1.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.25, val, fontsize=10, ha='center', va='center', color='#333')

x_ell = start_x + 4 * cell_w + 0.5
ax1.text(x_ell, start_y + cell_h/2, r"$\cdots$", fontsize=20, ha='center', va='center')

x_last_c = x_ell + 0.8
rect = patches.FancyBboxPatch((x_last_c + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=COLOR_CENTER, alpha=0.35)
ax1.add_patch(rect)
ax1.text(x_last_c + cell_w/2, start_y + cell_h*0.65, r"$c_D$", fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
ax1.text(x_last_c + cell_w/2, start_y + cell_h*0.25, "0.02", fontsize=10, ha='center', va='center', color='#333')

ax1.annotate('Centroide $c \\in \\mathbb{R}^D$', xy=(start_x + 2*cell_w, start_y), xytext=(start_x + 2*cell_w, start_y - 0.42), arrowprops=dict(arrowstyle='->', color=COLOR_CENTER, lw=1.5), fontsize=10, ha='center', color=COLOR_CENTER, fontweight='bold')

start_v = x_last_c + cell_w + 0.5
genes_v = [
    (r"$v_1$", "0.57", COLOR_V), (r"$v_2$", "-0.1", COLOR_V),
    (r"$v_3$", "0.80", COLOR_V),
]
for i, (lbl, val, col) in enumerate(genes_v):
    rect = patches.FancyBboxPatch((start_v + i * cell_w + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=col, alpha=0.35)
    ax1.add_patch(rect)
    ax1.text(start_v + i * cell_w + cell_w/2, start_y + cell_h*0.65, lbl, fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
    ax1.text(start_v + i * cell_w + cell_w/2, start_y + cell_h*0.25, val, fontsize=10, ha='center', va='center', color='#333')

x_ell_v = start_v + 3 * cell_w + 0.5
ax1.text(x_ell_v, start_y + cell_h/2, r"$\cdots$", fontsize=20, ha='center', va='center')

x_last_v = x_ell_v + 0.8
rect = patches.FancyBboxPatch((x_last_v + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=COLOR_V, alpha=0.35)
ax1.add_patch(rect)
ax1.text(x_last_v + cell_w/2, start_y + cell_h*0.65, r"$v_D$", fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
ax1.text(x_last_v + cell_w/2, start_y + cell_h*0.25, "0.15", fontsize=10, ha='center', va='center', color='#333')

ax1.annotate('Vector Principal $v \\in \\mathbb{R}^D$ (Eje de Proyección polar)', xy=(start_v + 2*cell_w, start_y), xytext=(start_v + 2*cell_w, start_y - 0.42), arrowprops=dict(arrowstyle='->', color=COLOR_V, lw=1.5), fontsize=10, ha='center', color=COLOR_V, fontweight='bold')
ax1.set_xlim(-0.5, 15)
ax1.set_ylim(start_y - 0.7, start_y + cell_h + 0.5)

# ─── FILA 2: Coeficientes de Chebyshev a_k ───────────────────────────
ax2 = axes[1]
genes_a = [
    (r"$a_0$", "1.50", COLOR_A), (r"$a_1$", "0.33", COLOR_A),
    (r"$a_2$", "-0.1", COLOR_A), (r"$a_3$", "0.05", COLOR_A),
]
for i, (lbl, val, col) in enumerate(genes_a):
    rect = patches.FancyBboxPatch((start_x + i * cell_w + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=col, alpha=0.35)
    ax2.add_patch(rect)
    ax2.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.65, lbl, fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
    ax2.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.25, val, fontsize=10, ha='center', va='center', color='#333')

x_ell_a = start_x + 4 * cell_w + 0.5
ax2.text(x_ell_a, start_y + cell_h/2, r"$\cdots$", fontsize=20, ha='center', va='center')

x_last_a = x_ell_a + 0.8
rect = patches.FancyBboxPatch((x_last_a + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=COLOR_A, alpha=0.35)
ax2.add_patch(rect)
ax2.text(x_last_a + cell_w/2, start_y + cell_h*0.65, r"$a_H$", fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
ax2.text(x_last_a + cell_w/2, start_y + cell_h*0.25, "0.01", fontsize=10, ha='center', va='center', color='#333')

ax2.annotate('Coeficientes serie de Cosenos (Chebyshev) $a_k \\in \\mathbb{R}^{H+1}$', xy=(start_x + 2.5*cell_w, start_y), xytext=(start_x + 2.5*cell_w, start_y - 0.42), arrowprops=dict(arrowstyle='->', color=COLOR_A, lw=1.5), fontsize=10, ha='center', color=COLOR_A, fontweight='bold')
ax2.set_xlim(-0.5, 15)
ax2.set_ylim(start_y - 0.7, start_y + cell_h + 0.5)

# ─── FILA 3: Pesos w (Feature Selection) ───────────────────────────
ax3 = axes[2]
genes_w = [
    (r"$w_1$", "0.99", COLOR_WEIGHTS), (r"$w_2$", "1.00", COLOR_WEIGHTS),
    (r"$w_3$", "0.01", COLOR_WEIGHTS), (r"$w_4$", "0.00", COLOR_WEIGHTS),
]
for i, (lbl, val, col) in enumerate(genes_w):
    rect = patches.FancyBboxPatch((start_x + i * cell_w + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=col, alpha=0.35)
    ax3.add_patch(rect)
    ax3.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.65, lbl, fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
    ax3.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.25, val, fontsize=10, ha='center', va='center', color='#333')

x_ell_w = start_x + 4 * cell_w + 0.5
ax3.text(x_ell_w, start_y + cell_h/2, r"$\cdots$", fontsize=20, ha='center', va='center')

x_last_w = x_ell_w + 0.8
rect = patches.FancyBboxPatch((x_last_w + 0.05, start_y), cell_w - 0.1, cell_h, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor='#555', facecolor=COLOR_WEIGHTS, alpha=0.35)
ax3.add_patch(rect)
ax3.text(x_last_w + cell_w/2, start_y + cell_h*0.65, r"$w_D$", fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
ax3.text(x_last_w + cell_w/2, start_y + cell_h*0.25, "0.00", fontsize=10, ha='center', va='center', color='#333')

ax3.annotate('Pesos de Características $w \\in [0,1]^D$ (Selección de variables por Lasso L1)', xy=(start_x + 2.5*cell_w, start_y), xytext=(start_x + 2.5*cell_w, start_y - 0.42), arrowprops=dict(arrowstyle='->', color=COLOR_WEIGHTS, lw=1.5), fontsize=10, ha='center', color=COLOR_WEIGHTS, fontweight='bold')
ax3.set_xlim(-0.5, 15)
ax3.set_ylim(start_y - 0.7, start_y + cell_h + 0.5)

plt.tight_layout()
plt.savefig('build/plot_angular_chromosome.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved build/plot_angular_chromosome.png")
