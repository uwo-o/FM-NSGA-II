import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# --- Layout: dos filas ---
# Fila 1: Centroide | Armónicos (k=1..H) | ...
# Fila 2: Pesos de Características w (nueva sección)

fig, axes = plt.subplots(2, 1, figsize=(14, 5.5))
fig.patch.set_facecolor('#F8F9FA')

for ax in axes:
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

cell_w = 1.0
cell_h = 0.55
start_x = 0.0
start_y = 0.2

# ─── Paleta de colores ────────────────────────────────────────
COLOR_CENTER  = "#E63946"   # rojo – centroide
COLOR_A       = "#2A9D8F"   # verde azulado – coefs coseno
COLOR_B       = "#E9C46A"   # amarillo – coefs seno
COLOR_WEIGHTS = "#6A4C93"   # violeta – pesos w (nuevo)

# ─── FILA 1: Centroide + Armónicos ───────────────────────────
ax1 = axes[0]
ax1.text(7.5, start_y + cell_h + 0.32,
         "Estructura del Cromosoma — Individuo FM-NSGA-II",
         fontsize=13, fontweight='bold', ha='center',
         color='#1D1D1D')

genes_row1 = [
    # (label, value, color)
    (r"$c_x$",       "0.06", COLOR_CENTER),
    (r"$c_y$",       "0.63", COLOR_CENTER),
    (r"$a_1^x$",     "0.49", COLOR_A),
    (r"$b_1^x$",     "0.46", COLOR_B),
    (r"$a_1^y$",     "0.23", COLOR_A),
    (r"$b_1^y$",    "-0.31", COLOR_B),
    (r"$a_2^x$",     "0.44", COLOR_A),
    (r"$b_2^x$",     "0.45", COLOR_B),
    (r"$a_2^y$",     "0.00", COLOR_A),
    (r"$b_2^y$",    "-0.31", COLOR_B),
]

for i, (lbl, val, col) in enumerate(genes_row1):
    rect = patches.FancyBboxPatch(
        (start_x + i * cell_w + 0.05, start_y), cell_w - 0.1, cell_h,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor='#555', facecolor=col, alpha=0.35)
    ax1.add_patch(rect)
    ax1.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.65,
             lbl, fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
    ax1.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.25,
             val, fontsize=10, ha='center', va='center', color='#333')

# Ellipsis
x_ell = start_x + len(genes_row1) * cell_w + 0.5
ax1.text(x_ell, start_y + cell_h/2, r"$\cdots$", fontsize=20, ha='center', va='center')

# Último armónico H
x_last = x_ell + 0.8
last = [(r"$a_H^x$","0.01",COLOR_A),(r"$b_H^x$","-0.02",COLOR_B),
        (r"$a_H^y$","0.00",COLOR_A),(r"$b_H^y$","0.03",COLOR_B)]
for j, (lbl, val, col) in enumerate(last):
    rect = patches.FancyBboxPatch(
        (x_last + j * cell_w + 0.05, start_y), cell_w - 0.1, cell_h,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor='#555', facecolor=col, alpha=0.35)
    ax1.add_patch(rect)
    ax1.text(x_last + j * cell_w + cell_w/2, start_y + cell_h*0.65,
             lbl, fontsize=12, ha='center', va='center', fontweight='bold', color='#1D1D1D')
    ax1.text(x_last + j * cell_w + cell_w/2, start_y + cell_h*0.25,
             val, fontsize=10, ha='center', va='center', color='#333')

# Anotaciones fila 1
ax1.annotate('Centroide\n$a_0 \\in \\mathbb{R}^D$',
             xy=(start_x + cell_w, start_y),
             xytext=(start_x + cell_w, start_y - 0.42),
             arrowprops=dict(arrowstyle='->', color=COLOR_CENTER, lw=1.5),
             fontsize=9, ha='center', color=COLOR_CENTER, fontweight='bold')

ax1.annotate('Armónico $k=1$\n$(a_1, b_1) \\in \\mathbb{R}^{2D}$',
             xy=(start_x + 5*cell_w, start_y),
             xytext=(start_x + 5*cell_w, start_y - 0.42),
             arrowprops=dict(arrowstyle='->', color='#444', lw=1.5),
             fontsize=9, ha='center', color='#444')

ax1.annotate('Armónico $k=H$\n$(a_H, b_H) \\in \\mathbb{R}^{2D}$',
             xy=(x_last + 2*cell_w, start_y),
             xytext=(x_last + 2*cell_w, start_y - 0.42),
             arrowprops=dict(arrowstyle='->', color='#444', lw=1.5),
             fontsize=9, ha='center', color='#444')

ax1.set_xlim(-0.5, x_last + 5)
ax1.set_ylim(start_y - 0.7, start_y + cell_h + 0.5)

# ─── FILA 2: Pesos de Características w (NUEVO) ───────────────
ax2 = axes[1]

ax2.text(7.5, start_y + cell_h + 0.32,
         r"Vector de Pesos de Características $\mathbf{w} \in [0,1]^D$ — Co-evolucionado con los coeficientes de Fourier",
         fontsize=11, fontweight='bold', ha='center', color=COLOR_WEIGHTS)

weights_labels = [r"$w_1$", r"$w_2$", r"$w_3$", r"$w_4$",
                  r"$w_5$", r"$w_6$", r"$w_7$", r"$w_8$"]
weights_values = ["0.98", "0.95", "0.03", "0.01",
                  "0.07", "0.02", "0.91", "0.88"]
# Los primeros, últimos y penúltimos son "relevantes" (valores altos)
# Los del medio son irrelevantes (valores bajos)

for i, (lbl, val) in enumerate(zip(weights_labels, weights_values)):
    wval = float(val)
    # Intensidad proporcional al peso
    alpha = 0.15 + 0.65 * wval
    rect = patches.FancyBboxPatch(
        (start_x + i * cell_w + 0.05, start_y), cell_w - 0.1, cell_h,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=COLOR_WEIGHTS, facecolor=COLOR_WEIGHTS, alpha=alpha)
    ax2.add_patch(rect)
    ax2.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.65,
             lbl, fontsize=12, ha='center', va='center', fontweight='bold',
             color='white' if alpha > 0.45 else '#444')
    ax2.text(start_x + i * cell_w + cell_w/2, start_y + cell_h*0.25,
             val, fontsize=10, ha='center', va='center',
             color='white' if alpha > 0.45 else '#444')

# Ellipsis
ax2.text(start_x + len(weights_labels) * cell_w + 0.5, start_y + cell_h/2,
         r"$\cdots$", fontsize=20, ha='center', va='center')

# Último peso
x_wlast = start_x + len(weights_labels) * cell_w + 1.2
rect = patches.FancyBboxPatch(
    (x_wlast + 0.05, start_y), cell_w - 0.1, cell_h,
    boxstyle="round,pad=0.02", linewidth=1.5,
    edgecolor=COLOR_WEIGHTS, facecolor=COLOR_WEIGHTS, alpha=0.80)
ax2.add_patch(rect)
ax2.text(x_wlast + cell_w/2, start_y + cell_h*0.65,
         r"$w_D$", fontsize=12, ha='center', va='center', fontweight='bold', color='white')
ax2.text(x_wlast + cell_w/2, start_y + cell_h*0.25,
         "0.89", fontsize=10, ha='center', va='center', color='white')

# Anotación explicativa
ax2.annotate(
    r'Dimensión informativa ($w_d \approx 1$)',
    xy=(start_x + 0.5, start_y),
    xytext=(start_x + 0.5, start_y - 0.42),
    arrowprops=dict(arrowstyle='->', color=COLOR_WEIGHTS, lw=1.5),
    fontsize=9, ha='center', color=COLOR_WEIGHTS, fontweight='bold')

ax2.annotate(
    r'Dimensión de ruido ($w_d \approx 0$)',
    xy=(start_x + 2.5*cell_w + 0.5, start_y),
    xytext=(start_x + 2.5*cell_w + 0.5, start_y - 0.42),
    arrowprops=dict(arrowstyle='->', color='#888', lw=1.5),
    fontsize=9, ha='center', color='#888')

ax2.set_xlim(-0.5, x_wlast + 2.5)
ax2.set_ylim(start_y - 0.7, start_y + cell_h + 0.5)

plt.tight_layout(pad=1.5)
plt.savefig("../build/plot_chromosome.png", dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Guardado: plot_chromosome.png")
