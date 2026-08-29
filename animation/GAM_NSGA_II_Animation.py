"""
GAM_NSGA_II_Animation.py — Animación de las partes nuevas del GAM Manifold
============================================================================
Cubre SOLO las escenas inéditas respecto a FM/AM:
  1. Portada: GAM-NSGA-II y su ecuación
  2. Cromosoma GAM  — β₀, w, {a_{ik}, b_{ik}} por feature
  3. Additividad     — descomposición visual f₁(x₁) + f₂(x₂)
  4. Interpretabilidad por feature — barras w_i · f_i(x_i)
  5. Frontera implícita — d(x)=0 en 2D, curvas de nivel
  6. Feature Selection — w → 0 apaga dimensiones
  7. Benchmark comparativo FM / AM / GAM en 3 datasets clave
  8. Outro / Tabla de propiedades

Renderizar:
    cd animation
    .venv/bin/manim -pqh GAM_NSGA_II_Animation.py GAMNSGAAnimation
"""

from manim import *
import numpy as np

# ─── Paleta de color ─────────────────────────────────────────────
C_BG    = "#0D0D1A"
C_TEXT  = "#E8E8F0"
C_DIM   = "#888899"
C_FM    = "#4CC9F0"   # azul Fourier
C_AM    = "#9D4EDD"   # violeta Angular
C_GAM   = "#2EC4B6"   # verde-cyan GAM
C_PLUS  = "#FF6B6B"   # clase positiva
C_NEG   = "#4CC9F0"   # clase negativa
C_W     = "#F77F00"   # peso w_i
C_BETA  = "#FFD166"   # β₀

# ─── Helpers matemáticos ─────────────────────────────────────────

def fourier_fi(x, coefs_a, coefs_b):
    """f_i(x) = Σ_k [ a_k cos(kπx) + b_k sin(kπx) ]"""
    v = 0.0
    H = len(coefs_a)
    for k in range(1, H + 1):
        v += coefs_a[k-1] * np.cos(k * np.pi * x) + coefs_b[k-1] * np.sin(k * np.pi * x)
    return v

def gam_decision(x1, x2, beta0, w1, a1, b1, w2, a2, b2):
    """d(x1,x2) = β₀ + w1·f1(x1) + w2·f2(x2)"""
    return beta0 + w1 * fourier_fi(x1, a1, b1) + w2 * fourier_fi(x2, a2, b2)

def make_two_moons(n=80, noise=0.07, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n // 2)
    x0 = np.column_stack([np.cos(t), np.sin(t)])
    x1 = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    x0 += rng.normal(0, noise, x0.shape)
    x1 += rng.normal(0, noise, x1.shape)
    pts = np.vstack([x0, x1])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    lo, hi = pts.min(0) - 0.1, pts.max(0) + 0.1
    pts = (pts - lo) / (hi - lo)
    return pts, labels


class GAMNSGAAnimation(Scene):
    def construct(self):
        self.camera.background_color = C_BG
        self._title()
        self._chromosome()
        self._additivity()
        self._feature_selection()
        self._decision_boundary()
        self._benchmark()
        self._outro()

    # ─── 1. PORTADA ──────────────────────────────────────────────
    def _title(self):
        title = Text("GAM-NSGA-II", font_size=64, weight=BOLD, color=C_GAM)
        sub   = Text("Modelo Aditivo Generalizado como Kernel Topológico",
                     font_size=26, color=C_TEXT)
        eq    = MathTex(
            r"d(\mathbf{x}) = \beta_0 + \sum_{i=1}^{D} w_i\, f_i(x_i)",
            font_size=38, color=WHITE
        )
        eq.set_color_by_tex(r"\beta_0", C_BETA)
        eq.set_color_by_tex(r"w_i", C_W)
        eq.set_color_by_tex(r"f_i", C_GAM)

        group = VGroup(title, sub, eq).arrange(DOWN, buff=0.55).move_to(ORIGIN)

        box = SurroundingRectangle(group, corner_radius=0.18,
                                   color=C_GAM, buff=0.35, stroke_width=2)
        box.set_fill(color=C_GAM, opacity=0.06)

        tag = VGroup(
            Text("Kernel 3 / RM-NSGA-II", font_size=16, color=C_DIM)
        ).to_corner(DR, buff=0.3)

        self.play(FadeIn(box), Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP*0.2), run_time=0.7)
        self.play(Write(eq), run_time=1.2)
        self.play(FadeIn(tag))
        self.wait(2.5)
        self.play(FadeOut(VGroup(box, title, sub, eq, tag)))

    # ─── 2. CROMOSOMA GAM ────────────────────────────────────────
    def _chromosome(self):
        title = Text("Cromosoma GAM", font_size=34, color=C_GAM).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # genes: β₀ | w₁ a₁₁ b₁₁ a₁₂ b₁₂ | w₂ a₂₁ b₂₁ a₂₂ b₂₂
        labels = [r"\beta_0",
                  "w_1", "a_{1,1}", "b_{1,1}", "a_{1,2}", "b_{1,2}",
                  "w_2", "a_{2,1}", "b_{2,1}", "a_{2,2}", "b_{2,2}"]
        colors = [C_BETA,
                  C_W, C_GAM, C_GAM, C_GAM, C_GAM,
                  C_W, C_GAM, C_GAM, C_GAM, C_GAM]
        W = 0.65
        row = VGroup()
        for lbl, col in zip(labels, colors):
            box = RoundedRectangle(width=W, height=0.55, corner_radius=0.08,
                                   fill_opacity=0.25, fill_color=col,
                                   stroke_color=col, stroke_width=1.8)
            lab = MathTex(lbl, font_size=18, color=WHITE)
            row.add(VGroup(box, lab))
        row.arrange(RIGHT, buff=0.06).shift(DOWN*1.8)

        # separadores entre bloques
        sep1 = DashedLine(
            row[0].get_right() + RIGHT*0.03 + UP*0.4,
            row[0].get_right() + RIGHT*0.03 + DOWN*0.4,
            color=C_DIM, stroke_width=1.5
        )
        sep2 = DashedLine(
            row[5].get_right() + RIGHT*0.03 + UP*0.4,
            row[5].get_right() + RIGHT*0.03 + DOWN*0.4,
            color=C_DIM, stroke_width=1.5
        )

        # braces
        brace_b  = Brace(row[0], DOWN, buff=0.05, color=C_BETA)
        brace_f1 = Brace(VGroup(*list(row)[1:6]), DOWN, buff=0.05, color=C_GAM)
        brace_f2 = Brace(VGroup(*list(row)[6:]),  DOWN, buff=0.05, color=C_GAM)

        bt_b  = Tex("intercepto", font_size=14).set_color(C_BETA); brace_b.put_at_tip(bt_b)
        bt_f1 = Tex("feature 1 ($w, H=2$)", font_size=14).set_color(C_GAM); brace_f1.put_at_tip(bt_f1)
        bt_f2 = Tex("feature 2 ($w, H=2$)", font_size=14).set_color(C_GAM); brace_f2.put_at_tip(bt_f2)

        size_note = Text(
            "Tamaño: 1 + D·(1 + 2H)   —   escala lineal en D",
            font_size=17, color=C_DIM
        ).to_edge(DOWN, buff=0.25)

        self.play(LaggedStart(*[GrowFromCenter(c) for c in row], lag_ratio=0.06))
        self.play(FadeIn(sep1, sep2))
        self.play(FadeIn(brace_b, bt_b, brace_f1, bt_f1, brace_f2, bt_f2))
        self.play(Write(size_note))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, row, sep1, sep2,
                                 brace_b, bt_b, brace_f1, bt_f1, brace_f2, bt_f2,
                                 size_note)))

    # ─── 3. ADDITIVIDAD ──────────────────────────────────────────
    def _additivity(self):
        title = Text("Estructura Aditiva", font_size=34, color=C_GAM).to_edge(UP, buff=0.3)
        eq_full = MathTex(
            r"d(\mathbf{x}) = \beta_0 + w_1 f_1(x_1) + w_2 f_2(x_2)",
            font_size=30, color=WHITE
        ).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), Write(eq_full))

        # Axes para f1 y f2
        ax_kw = dict(x_range=[0, 1, 0.25], y_range=[-1.5, 1.5, 0.5],
                     x_length=3.5, y_length=2.5,
                     axis_config={"color": C_DIM, "stroke_width": 1.5},
                     tips=False)

        ax1 = Axes(**ax_kw).shift(LEFT*3.3 + DOWN*0.8)
        ax2 = Axes(**ax_kw).shift(RIGHT*0.7 + DOWN*0.8)

        lbl1 = MathTex(r"f_1(x_1)", font_size=22, color=C_FM).next_to(ax1, UP, buff=0.08)
        lbl2 = MathTex(r"f_2(x_2)", font_size=22, color=C_AM).next_to(ax2, UP, buff=0.08)

        # coeficientes de ejemplo
        a1 = [0.6, -0.3]; b1 = [0.4, 0.2]
        a2 = [-0.5, 0.4]; b2 = [0.3, -0.1]

        xs = np.linspace(0, 1, 200)
        f1_vals = np.array([fourier_fi(x, a1, b1) for x in xs])
        f2_vals = np.array([fourier_fi(x, a2, b2) for x in xs])

        def pts_from_vals(ax, vals):
            return [ax.c2p(xs[i], vals[i]) for i in range(len(xs))]

        curve1 = VMobject(stroke_color=C_FM, stroke_width=2.5)
        curve1.set_points_smoothly(pts_from_vals(ax1, f1_vals))
        curve2 = VMobject(stroke_color=C_AM, stroke_width=2.5)
        curve2.set_points_smoothly(pts_from_vals(ax2, f2_vals))

        plus = MathTex(r"+", font_size=48, color=C_TEXT).move_to(
            (ax1.get_right() + ax2.get_left()) / 2
        )

        # Ejes y curvas
        self.play(FadeIn(ax1, ax2, lbl1, lbl2))
        self.play(Create(curve1), Create(curve2), FadeIn(plus))

        # Muestra un punto x* y su descomposición
        x_star = 0.6
        f1_star = fourier_fi(x_star, a1, b1)
        f2_star = fourier_fi(x_star, a2, b2)
        beta0   = 0.1
        d_star  = beta0 + 0.9 * f1_star + 0.8 * f2_star

        dot1 = Dot(ax1.c2p(x_star, f1_star), color=C_FM, radius=0.1)
        dot2 = Dot(ax2.c2p(x_star, f2_star), color=C_AM, radius=0.1)

        val_text = MathTex(
            rf"d = {beta0:.2f} + 0.9 \cdot({f1_star:.2f}) + 0.8 \cdot({f2_star:.2f}) = {d_star:.3f}",
            font_size=22, color=C_TEXT
        ).to_edge(DOWN, buff=0.3)

        sign_text = Text(
            "d < 0  →  clase positiva   |   d ≥ 0  →  clase negativa",
            font_size=18, color=C_DIM
        ).next_to(val_text, UP, buff=0.15)

        self.play(FadeIn(dot1, dot2))
        self.play(Write(sign_text), Write(val_text))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, eq_full, ax1, ax2, lbl1, lbl2,
                                 curve1, curve2, plus, dot1, dot2,
                                 val_text, sign_text)))

    # ─── 4. FEATURE SELECTION (w_i → 0) ─────────────────────────
    def _feature_selection(self):
        title = Text("Feature Selection Automático", font_size=34, color=C_GAM)\
            .to_edge(UP, buff=0.3)
        sub = Text("w_i → 0 apaga la feature i sin post-procesamiento",
                   font_size=20, color=C_DIM).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(sub))

        # Barras de importancia: 5 features
        feature_names = ["Radius", "Texture", "Perimeter", "Area", "Smoothness"]
        initial_w = [0.92, 0.35, 0.78, 0.55, 0.12]
        final_w   = [0.92, 0.04, 0.80, 0.52, 0.02]   # evolución: 1 y 4 desaparecen
        colors_bar = [C_GAM if w > 0.2 else "#E63946" for w in initial_w]

        ax = Axes(
            x_range=[0, 1, 0.25], y_range=[0, 5, 1],
            x_length=5, y_length=3.5,
            axis_config={"color": C_DIM, "stroke_width": 1.5},
            tips=False
        ).shift(LEFT*1.5 + DOWN*0.6)

        ax_lbl = ax.get_x_axis_label(MathTex(r"w_i", font_size=22, color=C_W))

        bars = VGroup()
        bar_labels = VGroup()
        for i, (w, name) in enumerate(zip(initial_w, feature_names)):
            y_pos = ax.c2p(0, i + 0.5)[1]
            x_start = ax.c2p(0, 0)[0]
            x_end   = ax.c2p(w, 0)[0]
            bar = Rectangle(
                width=x_end - x_start, height=0.38,
                fill_color=C_GAM, fill_opacity=0.85,
                stroke_width=0
            ).align_to(ax.c2p(0, i + 0.15), LEFT + DOWN)
            lbl = Text(name, font_size=15, color=C_TEXT)\
                .next_to(ax.c2p(0, i + 0.5), LEFT, buff=0.15)
            bars.add(bar)
            bar_labels.add(lbl)

        self.play(FadeIn(ax, ax_lbl), FadeIn(bar_labels))
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1))

        # Etiqueta "NSGA-II evoluciona →"
        arrow_note = Text("NSGA-II evoluciona →", font_size=17, color=C_DIM)\
            .to_edge(RIGHT, buff=1.2).shift(UP*1.5)

        self.play(FadeIn(arrow_note))
        self.wait(0.8)

        # Animar cambio de w inicial → w final
        anims = []
        for i, (bar, w_new) in enumerate(zip(bars, final_w)):
            x_start = ax.c2p(0, 0)[0]
            x_end   = ax.c2p(w_new, 0)[0]
            new_width = max(x_end - x_start, 0.02)
            new_color = "#E63946" if w_new < 0.1 else C_GAM
            target = bar.copy()
            target.stretch_to_fit_width(new_width)
            target.align_to(ax.c2p(0, i + 0.15), LEFT + DOWN)
            target.set_fill(color=new_color, opacity=0.85)
            anims.append(Transform(bar, target))

        self.play(*anims, run_time=1.8)

        # Notas finales
        note = VGroup(
            Text("w ≈ 0.04  →  feature eliminada   (sin SHAP, sin LIME)", font_size=16, color="#E63946"),
            Text("El genotipo evoluciona hacia escasez naturalmente", font_size=16, color=C_DIM),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, sub, ax, ax_lbl, bars,
                                 bar_labels, arrow_note, note)))

    # ─── 5. FRONTERA DE DECISIÓN d(x)=0 ─────────────────────────
    def _decision_boundary(self):
        title = Text("Frontera de Decisión Implícita  d(x) = 0",
                     font_size=30, color=C_GAM).to_edge(UP, buff=0.3)
        sub = MathTex(
            r"d(x_1, x_2) = \beta_0 + w_1 f_1(x_1) + w_2 f_2(x_2)",
            font_size=22, color=C_DIM
        ).next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(sub))

        # Dataset Two-Moons
        pts, labels = make_two_moons(n=100, noise=0.08)

        ax = Axes(
            x_range=[0, 1, 0.25], y_range=[0, 1, 0.25],
            x_length=5, y_length=5,
            axis_config={"color": C_DIM, "stroke_width": 1.2},
            tips=False
        ).shift(DOWN*0.5)

        # Puntos
        dots = VGroup()
        for p, lbl in zip(pts, labels):
            col = C_PLUS if lbl == 0 else C_NEG
            dots.add(Dot(ax.c2p(p[0], p[1]), radius=0.07,
                         color=col, fill_opacity=0.85))

        # GAM aprendido para Two-Moons (coeficientes manuales razonables)
        beta0 = -0.15
        w1, w2 = 0.9, 0.8
        a1, b1 = [0.5, -0.3], [0.4, 0.2]
        a2, b2 = [-0.6, 0.35], [0.3, -0.15]

        # Contorno d(x)=0
        N = 100
        xs_g = np.linspace(0, 1, N)
        ys_g = np.linspace(0, 1, N)
        Z = np.array([[gam_decision(x, y, beta0, w1, a1, b1, w2, a2, b2)
                       for x in xs_g] for y in ys_g])

        # Dibujar la curva de nivel 0 manualmente (marching-squares simplificado)
        boundary_pts = []
        for i in range(N - 1):
            for j in range(N - 1):
                # Cuadrícula: buscar cambios de signo
                vals = [Z[i,j], Z[i,j+1], Z[i+1,j], Z[i+1,j+1]]
                if min(vals) < 0 < max(vals):
                    cx = (xs_g[j] + xs_g[j+1]) / 2
                    cy = (ys_g[i] + ys_g[i+1]) / 2
                    boundary_pts.append(ax.c2p(cx, cy))

        if len(boundary_pts) > 10:
            # Ordenar aproximadamente por ángulo desde el centroide
            arr = np.array([[p[0], p[1]] for p in boundary_pts])
            cx_m, cy_m = arr.mean(axis=0)
            angles = np.arctan2(arr[:, 1] - cy_m, arr[:, 0] - cx_m)
            idx = np.argsort(angles)
            boundary_pts = [boundary_pts[i] for i in idx]
            boundary_curve = VMobject(stroke_color=C_GAM, stroke_width=3,
                                      fill_opacity=0)
            boundary_curve.set_points_smoothly(boundary_pts)
        else:
            boundary_curve = VMobject()  # vacío si no hay suficientes puntos

        neg_label = Text("Clase −", font_size=15, color=C_NEG).to_corner(DR, buff=1.0)
        pos_label = Text("Clase +", font_size=15, color=C_PLUS).to_corner(DL, buff=1.0)

        self.play(FadeIn(ax))
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.02))
        self.play(Create(boundary_curve), run_time=1.5)
        self.play(FadeIn(neg_label, pos_label))

        note = Text(
            "La frontera NO es cerrada — diferencia clave vs FM/AM",
            font_size=17, color="#FFD166"
        ).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, sub, ax, dots, boundary_curve,
                                 neg_label, pos_label, note)))

    # ─── 6. BENCHMARK COMPARATIVO ────────────────────────────────
    def _benchmark(self):
        title = Text("Resultados — 3 Datasets Clave", font_size=32, color=C_GAM)\
            .to_edge(UP, buff=0.3)
        self.play(Write(title))

        datasets = ["Circles 2D", "Adversarial 52D", "Breast Cancer 30D"]
        fm_vals  = [0.8750, 0.6125, 0.9292]
        am_vals  = [0.4375, 0.4500, 0.9381]
        gam_vals = [0.9875, 0.6375, 0.9735]

        W_BAR = 0.22
        SCALE = 4.5

        def make_group(ds_name, fm, am, gam, x_offset):
            bars_g = VGroup()
            for val, col, lbl in [(fm, C_FM, "FM"), (am, C_AM, "AM"), (gam, C_GAM, "GAM")]:
                bar = Rectangle(
                    width=W_BAR, height=val * SCALE,
                    fill_color=col, fill_opacity=0.85, stroke_width=0
                )
                val_t = Text(f"{val:.3f}", font_size=12, color=WHITE)\
                    .next_to(bar, UP, buff=0.05)
                name_t = Text(lbl, font_size=12, color=col)\
                    .next_to(bar, DOWN, buff=0.05)
                bars_g.add(VGroup(bar, val_t, name_t))
            bars_g.arrange(RIGHT, buff=0.08)
            # Alinear bases
            ref_y = bars_g[0][0].get_bottom()[1]
            for bg in bars_g:
                bg[0].align_to(bars_g[0][0], DOWN)

            ds_lbl = Text(ds_name, font_size=14, color=C_TEXT)\
                .next_to(bars_g, DOWN, buff=0.2)
            return VGroup(bars_g, ds_lbl).shift(RIGHT * x_offset)

        groups = VGroup()
        for i, (ds, fm, am, gam) in enumerate(zip(datasets, fm_vals, am_vals, gam_vals)):
            g = make_group(ds, fm, am, gam, x_offset=(i - 1) * 3.3)
            groups.add(g)
        groups.shift(DOWN * 0.5)

        # Eje Y de referencia
        y_axis = Line(LEFT*5.2 + DOWN*1.8, LEFT*5.2 + UP*2.2,
                      color=C_DIM, stroke_width=1)
        for acc in [0.5, 0.75, 1.0]:
            tick_y = groups[0][0][0][0].get_bottom()[1] + acc * SCALE
            tick = Line(LEFT*5.3 + UP*(tick_y - 2), LEFT*5.1 + UP*(tick_y - 2),
                        color=C_DIM, stroke_width=1)
            lbl  = Text(f"{acc:.2f}", font_size=11, color=C_DIM)\
                .next_to(tick, LEFT, buff=0.05)
            groups.add(VGroup(tick, lbl))

        # Leyenda
        legend = VGroup(
            VGroup(Square(side_length=0.18, fill_color=C_FM,  fill_opacity=0.85, stroke_width=0),
                   Text("FM-NSGA-II",  font_size=14, color=C_FM )).arrange(RIGHT, buff=0.1),
            VGroup(Square(side_length=0.18, fill_color=C_AM,  fill_opacity=0.85, stroke_width=0),
                   Text("AM-NSGA-II",  font_size=14, color=C_AM )).arrange(RIGHT, buff=0.1),
            VGroup(Square(side_length=0.18, fill_color=C_GAM, fill_opacity=0.85, stroke_width=0),
                   Text("GAM-NSGA-II", font_size=14, color=C_GAM)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.5).to_corner(UR, buff=0.4)

        self.play(FadeIn(groups, legend))

        # Highlight GAM en Breast Cancer
        bc_gam_bar = groups[2][0][2][0]
        self.play(Flash(bc_gam_bar, color=C_GAM, line_length=0.15, num_lines=10))
        note = Text("GAM: 97.35% en Breast Cancer 30D  🏆",
                    font_size=17, color=C_GAM).to_edge(DOWN, buff=0.25)
        self.play(Write(note))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, groups, legend, note)))

    # ─── 7. OUTRO ────────────────────────────────────────────────
    def _outro(self):
        title = Text("GAM-NSGA-II — Propiedades", font_size=36, color=C_GAM)\
            .to_edge(UP, buff=0.4)
        self.play(Write(title))

        props = [
            ("✓ Interpretabilidad por feature", C_GAM, "  (sin SHAP, sin LIME)"),
            ("✓ Feature Selection intrínseco",  C_GAM, "  (w_i → 0)"),
            ("✓ Escalabilidad lineal en D",      C_GAM, "  O(H·D)"),
            ("✓ White-Box completo",             C_GAM, "  ecuación analítica"),
            ("✗ Sin frontera cerrada",           "#E63946", "  no aditivo (Two-Moons)"),
            ("✗ Sin interacciones x_i × x_j",   "#E63946", "  limitación estructural"),
        ]

        items = VGroup()
        for main, col, detail in props:
            line = VGroup(
                Text(main, font_size=20, color=col),
                Text(detail, font_size=16, color=C_DIM),
            ).arrange(RIGHT, buff=0.05)
            items.add(line)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN*0.3)

        self.play(LaggedStart(*[FadeIn(it, shift=RIGHT*0.2) for it in items],
                              lag_ratio=0.15))

        final = Text("Parte de la familia RM-NSGA-II (FM · AM · GAM)",
                     font_size=18, color=C_DIM).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(final))
        self.wait(3.0)
        self.play(FadeOut(VGroup(title, items, final)))
