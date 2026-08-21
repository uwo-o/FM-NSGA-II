"""
FM_NSGA_II_Animation.py — Manim Community Edition
===================================================
Renderizar con:
    manim -pqh FM_NSGA_II_Animation.py FMNSGAFullAnimation

Escenas:
  1. Portada
  2. Dataset Two-Moons
  3. Cromosoma de Fourier
  4. Población inicial aleatoria
  5. Evaluación de fitness (O1, O2)
  6. Frente de Pareto / selección
  7. Evolución generacional (morphing)
  8. Manifolds convergidos
  9. Inferencia: proyección + distancia radial + clase
 10. Resumen / Outro
"""

from manim import *
import numpy as np

# ─── Paleta ───────────────────────────────────────────────────────────────────
C_BG     = "#0D1117"
C_C0     = "#2EC4B6"
C_C1     = "#FF6B6B"
C_PARETO = "#F7C59F"
C_NEW    = "#FFD166"
C_TEXT   = "#E8E8E8"
C_DIM    = "#555D6B"
C_ACC    = "#9B5DE5"
C_OBJ1   = "#06D6A0"
C_OBJ2   = "#EF476F"

# ─── Datos sintéticos ─────────────────────────────────────────────────────────
def two_moons(n=40, noise=0.08, seed=42):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    t0 = rng.uniform(0, np.pi, n2)
    t1 = rng.uniform(0, np.pi, n2)
    c0 = np.stack([np.cos(t0), np.sin(t0)], 1)
    c1 = np.stack([1 - np.cos(t1), 0.5 - np.sin(t1)], 1)
    c0 += rng.normal(0, noise, c0.shape)
    c1 += rng.normal(0, noise, c1.shape)
    return c0, c1

# ─── Fourier helpers ──────────────────────────────────────────────────────────
def fourier_xy(center, A, B, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    H = A.shape[0]
    xy = np.tile(np.array(center, float), (n, 1))
    for k in range(1, H + 1):
        xy[:, 0] += A[k-1, 0]*np.cos(k*t) + B[k-1, 0]*np.sin(k*t)
        xy[:, 1] += A[k-1, 1]*np.cos(k*t) + B[k-1, 1]*np.sin(k*t)
    return xy

def make_curve(ax, center, A, B, color, stroke=3):
    pts = fourier_xy(center, A, B)
    p3d = [ax.c2p(p[0], p[1]) for p in pts]
    m = VMobject(color=color, stroke_width=stroke)
    m.set_points_as_corners(p3d + [p3d[0]])
    return m

def radial_ds(pt, center, A, B):
    pts  = fourier_xy(center, A, B, 300)
    idx  = np.argmin(np.linalg.norm(pts - np.array(pt), axis=1))
    g    = pts[idx]
    rx   = np.linalg.norm(np.array(pt) - np.array(center))
    rg   = np.linalg.norm(g            - np.array(center))
    return rx - rg, g

# ─── Manifolds analíticos (H=4) ──────────────────────────────────────────────
# Clase 0 — luna superior: media-disk radio≈1, centrada en (0, 0).
# Frontera = semicírculo superior + base plana.
# Coeficientes Fourier analíticos de max(0,sin(t))  × escala 1.18
_S = 1.18   # escala (padding de ruido)

C0 = np.array([0.0,  0.32])
A0 = np.array([
    [ _S,   0.00],            # a1x = R,  a1y = 0
    [ 0.00, -2*_S/(3*np.pi)], # a2x = 0,  a2y = -2R/(3π)  ← aplanamiento inferior
    [ 0.00,  0.00],
    [ 0.00,  2*_S/(15*np.pi)],# a4x = 0,  a4y = 2R/(15π)  ← refinamiento
])
B0 = np.array([
    [0.00,  _S/2],             # b1x = 0,  b1y = R/2  ← amplitud vertical
    [0.00,  0.00],
    [0.00,  0.00],
    [0.00,  0.00],
])

# Clase 1 — luna inferior: reflejo exacto de clase 0 por (x,y)→(1−x, 0.5−y)
C1 = np.array([1.0,  0.5 - C0[1]])
A1 = np.array([
    [-A0[0,0],  -A0[0,1]],
    [-A0[1,0],  -A0[1,1]],
    [-A0[2,0],  -A0[2,1]],
    [-A0[3,0],  -A0[3,1]],
])
B1 = np.array([
    [-B0[0,0],  -B0[0,1]],
    [-B0[1,0],  -B0[1,1]],
    [-B0[2,0],  -B0[2,1]],
    [-B0[3,0],  -B0[3,1]],
])

# Curvas aleatorias iniciales
rng0 = np.random.default_rng(7)
RAND_CURVES = [(rng0.uniform([-0.2, -0.2], [1.2, 0.8]),
                rng0.uniform(-0.6, 0.6, (2, 2)),
                rng0.uniform(-0.6, 0.6, (2, 2))) for _ in range(6)]


# ─── Datos 3D sintéticos y Helpers ────────────────────────────────────────────
C_C2 = "#FFE66D"

def blobs_3d(n=40, seed=42):
    rng = np.random.default_rng(seed)
    c0 = rng.normal([0, 1.5, 0], 0.35, (n, 3))
    c1 = rng.normal([1.5, -0.5, 1.5], 0.35, (n, 3))
    c2 = rng.normal([-1.5, -0.5, -1.5], 0.35, (n, 3))
    return c0, c1, c2

def fourier_xyz(center, A, B, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    H = A.shape[0]
    xyz = np.tile(np.array(center, float), (n, 1))
    for k in range(1, H + 1):
        xyz[:, 0] += A[k-1, 0]*np.cos(k*t) + B[k-1, 0]*np.sin(k*t)
        xyz[:, 1] += A[k-1, 1]*np.cos(k*t) + B[k-1, 1]*np.sin(k*t)
        xyz[:, 2] += A[k-1, 2]*np.cos(k*t) + B[k-1, 2]*np.sin(k*t)
    return xyz

def make_curve_3d(ax, center, A, B, color, stroke=4):
    pts = fourier_xyz(center, A, B)
    p3d = [ax.c2p(p[0], p[1], p[2]) for p in pts]
    m = VMobject(color=color, stroke_width=stroke)
    m.set_points_as_corners(p3d + [p3d[0]])
    return m

# Coeficientes para envolver cada blob en 3D
C0_3 = np.array([0, 1.5, 0])
A0_3 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
B0_3 = np.array([[0, 0, 1.0], [1.0, 0, 0]])

C1_3 = np.array([1.5, -0.5, 1.5])
A1_3 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
B1_3 = np.array([[0, 0, 1.0], [1.0, 0, 0]])

C2_3 = np.array([-1.5, -0.5, -1.5])
A2_3 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
B2_3 = np.array([[0, 0, 1.0], [1.0, 0, 0]])


# ══════════════════════════════════════════════════════════════════════════════
class FMNSGAFullAnimation(ThreeDScene):

    def construct(self):
        self.camera.background_color = C_BG
        self._title()
        self._dataset()
        self._chromosome()
        self._init_pop()
        self._fitness()
        self._pareto()
        self._evolution()
        self._final_manifold()
        self._scene_3d()
        self._inference()
        self._outro()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _hdr(self, txt):
        return Text(txt, font_size=30, color=C_ACC, weight=BOLD).to_edge(UP, buff=0.2)

    def _std_axes(self, w=10, h=6):
        return Axes(
            x_range=[-1.3, 2.3, 0.5], y_range=[-0.8, 1.3, 0.5],
            x_length=w, y_length=h,
            axis_config={"color": C_DIM, "include_tip": False},
        ).shift(DOWN * 0.2)

    # ── 1. Portada ────────────────────────────────────────────────────────────
    def _title(self):
        title = VGroup(
            Text("FM-NSGA-II", font_size=76, weight=BOLD, color=C_C0),
            Text("Fourier Manifold · Non-Dominated Sorting GA", font_size=28, color=C_TEXT),
            Text("Clasificación No-Lineal mediante Variedades Paramétricas", font_size=24, color=C_DIM),
        ).arrange(DOWN, buff=0.45).center()

        self.play(Write(title[0]), run_time=1.5)
        self.play(FadeIn(title[1], title[2]), run_time=1)
        self.wait(2)
        self.play(FadeOut(title))

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    def _dataset(self):
        hdr = self._hdr("Paso 1 — El Problema: Clasificación No Lineal")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, d1 = two_moons(n=50)
        dots0 = VGroup(*[Dot(ax.c2p(x,y), radius=0.09, color=C_C0) for x,y in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y), radius=0.09, color=C_C1) for x,y in d1])
        self.play(Create(ax))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots0], lag_ratio=0.04))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots1], lag_ratio=0.04))
        q = Text("¿Cómo separar estas lunas con una frontera continua?",
                 font_size=24, color=C_PARETO).to_edge(DOWN, buff=0.3)
        self.play(Write(q))
        self.wait(2)
        self.play(FadeOut(VGroup(hdr, ax, dots0, dots1, q)))

    # ── 3. Cromosoma ─────────────────────────────────────────────────────────
    def _chromosome(self):
        hdr = self._hdr("Paso 2 — El Cromosoma de Fourier")
        self.play(FadeIn(hdr))

        eq = MathTex(
            r"f_d(t) = \underbrace{a_{0,d}}_{\text{centroide}} "
            r"+ \sum_{k=1}^{H}\!\left(a_{k,d}\cos(kt)+b_{k,d}\sin(kt)\right)",
            font_size=36, color=C_TEXT
        ).shift(UP*2.5)
        self.play(Write(eq), run_time=2)

        # Celdas del cromosoma — fila 1: coeficientes
        specs1 = [
            (r"c_x",        "#E63946"),
            (r"c_y",        "#E63946"),
            (r"a_1^x",      "#2A9D8F"),
            (r"b_1^x",      "#E9C46A"),
            (r"a_1^y",      "#2A9D8F"),
            (r"b_1^y",      "#E9C46A"),
            (r"\cdots",     C_DIM),
            (r"a_H^x",      "#2A9D8F"),
            (r"b_H^y",      "#E9C46A"),
        ]
        # fila 2: pesos w
        specs2 = [(r"w_1", C_ACC), (r"w_2", C_ACC), (r"\cdots", C_DIM), (r"w_D", C_ACC)]

        def make_row(specs, y_off):
            row = VGroup()
            for lbl, col in specs:
                box = Rectangle(width=0.88, height=0.58,
                                fill_opacity=0.3, fill_color=col,
                                stroke_color=col, stroke_width=2)
                lab = MathTex(lbl, font_size=20, color=WHITE)
                row.add(VGroup(box, lab))
            row.arrange(RIGHT, buff=0.04)
            row.shift(DOWN * y_off)
            return row

        row1 = make_row(specs1, 0.0)
        row2 = make_row(specs2, 0.85)

        lbl1 = Text("Coeficientes Fourier", font_size=17, color=C_TEXT).next_to(row1, LEFT, buff=0.15)
        lbl2 = Text("Pesos  w  (nuevo)", font_size=17, color=C_ACC).next_to(row2, LEFT, buff=0.15)

        self.play(LaggedStart(*[GrowFromCenter(c) for c in row1], lag_ratio=0.07))
        self.play(Write(lbl1))
        self.play(LaggedStart(*[GrowFromCenter(c) for c in row2], lag_ratio=0.1))
        self.play(Write(lbl2))

        w_info = Text(
            "w co-evoluciona junto con los armónicos → aprende qué dimensiones importan",
            font_size=19, color=C_ACC
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(w_info))
        self.wait(2)
        self.play(FadeOut(VGroup(hdr, eq, row1, row2, lbl1, lbl2, w_info)))

    # ── 4. Población inicial ──────────────────────────────────────────────────
    def _init_pop(self):
        hdr = self._hdr("Paso 3 — Población Inicial Aleatoria (60 individuos)")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, _ = two_moons(n=40)
        dots = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C0, fill_opacity=0.5) for x,y in d0])
        self.play(Create(ax), FadeIn(dots))

        cols = [C_C0, C_C1, C_PARETO, C_ACC, "#FF9F1C", "#80FFDB"]
        mobs = VGroup()
        for i, (c, a, b) in enumerate(RAND_CURVES):
            m = make_curve(ax, c, a, b, cols[i], stroke=2)
            m.set_opacity(0.65)
            mobs.add(m)

        self.play(LaggedStart(*[Create(m) for m in mobs], lag_ratio=0.2), run_time=2.5)
        bad = Text("Fitness alto — clasificación pésima (O₁ ≈ 0.5)",
                   font_size=22, color=C_OBJ2).to_edge(DOWN, buff=0.3)
        self.play(Write(bad))
        self.play(Indicate(mobs, color=C_OBJ2))
        self.wait(1.5)
        self.play(FadeOut(VGroup(hdr, ax, dots, mobs, bad)))

    # ── 5. Fitness ────────────────────────────────────────────────────────────
    def _fitness(self):
        hdr = self._hdr("Paso 4 — Evaluación Bi-Objetivo de Fitness")
        self.play(FadeIn(hdr))
        ax = Axes(x_range=[-0.4,1.4,0.5], y_range=[-0.5,0.9,0.5],
                  x_length=5.5, y_length=4.2,
                  axis_config={"color": C_DIM, "include_tip": False}
                  ).to_edge(LEFT, buff=0.7).shift(DOWN*0.2)

        d0, _ = two_moons(n=40)
        dots = VGroup(*[Dot(ax.c2p(x,y), radius=0.08, color=C_C0) for x,y in d0])
        curve = make_curve(ax, C0, A0, B0, C_C0, stroke=3)

        self.play(Create(ax), FadeIn(dots), Create(curve))

        # Panel derecho
        o1 = VGroup(
            Text("O₁ — Error de clasificación", font_size=21, color=C_OBJ1, weight=BOLD),
            MathTex(r"O_1 = \frac{1}{N}\sum_i \mathbb{I}[y_i \neq \hat y_i]", font_size=26, color=C_TEXT),
            Text("→ minimizar", font_size=18, color=C_OBJ1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        o2 = VGroup(
            Text("O₂ — Complejidad topológica", font_size=21, color=C_OBJ2, weight=BOLD),
            MathTex(r"O_2 = \lambda\frac{L(M)}{L_0}+(1-\lambda)\frac{H}{H_{\max}}", font_size=26, color=C_TEXT),
            Text("→ minimizar (Navaja de Ockham)", font_size=18, color=C_OBJ2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        panel = VGroup(o1, Line(ORIGIN, RIGHT*4.5, color=C_DIM, stroke_width=1), o2
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.3
                       ).to_edge(RIGHT, buff=0.5).shift(DOWN*0.3)

        self.play(FadeIn(panel), run_time=1.2)
        self.wait(2)
        self.play(FadeOut(VGroup(hdr, ax, dots, curve, panel)))

    # ── 6. Pareto ─────────────────────────────────────────────────────────────
    def _pareto(self):
        hdr = self._hdr("Paso 5 — Selección NSGA-II: Frente de Pareto")
        self.play(FadeIn(hdr))

        ax = Axes(x_range=[0,1,0.2], y_range=[0,1,0.2],
                  x_length=7, y_length=5,
                  axis_config={"color": C_DIM, "include_tip": True}
                  ).shift(DOWN*0.3)
        xl = Text("O₁ (error)", font_size=22, color=C_OBJ1).next_to(ax.x_axis, DOWN, buff=0.3)
        yl = Text("O₂ (complejidad)", font_size=22, color=C_OBJ2
                  ).next_to(ax.y_axis, LEFT, buff=0.15).rotate(90*DEGREES)

        rng_p = np.random.default_rng(13)
        pts = rng_p.uniform([0.05,0.05], [0.95,0.95], (35, 2))
        dominated = []
        for i in range(len(pts)):
            dom = any(pts[j,0] <= pts[i,0] and pts[j,1] <= pts[i,1]
                      and (pts[j,0] < pts[i,0] or pts[j,1] < pts[i,1])
                      for j in range(len(pts)) if j != i)
            dominated.append(dom)

        dots_all = VGroup(*[Dot(ax.c2p(x,y), radius=0.09,
                                color=C_DIM if dominated[i] else C_PARETO)
                             for i, (x,y) in enumerate(pts)])

        self.play(Create(ax), Write(xl), Write(yl))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots_all], lag_ratio=0.04), run_time=2)

        front_pts = sorted([(pts[i,0], pts[i,1]) for i in range(len(pts)) if not dominated[i]])
        front = VMobject(color=C_PARETO, stroke_width=3)
        front.set_points_as_corners([ax.c2p(x,y) for x,y in front_pts])

        flbl = Text("Frente de Pareto — individuos no-dominados",
                    font_size=22, color=C_PARETO).to_edge(UP, buff=1.2)
        self.play(Create(front), Write(flbl))
        self.play(*[dots_all[i].animate.set_opacity(0.2) for i in range(len(pts)) if dominated[i]])
        info = Text("NSGA-II mantiene diversidad con Crowding Distance",
                    font_size=20, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.play(Write(info))
        self.wait(2.5)
        self.play(FadeOut(VGroup(hdr, ax, xl, yl, dots_all, front, flbl, info)))

    # ── 7. Evolución ──────────────────────────────────────────────────────────
    def _evolution(self):
        hdr = self._hdr("Paso 6 — Evolución Generacional")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, _ = two_moons(n=40)
        dots = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C0, fill_opacity=0.6) for x,y in d0])
        self.play(Create(ax), FadeIn(dots))

        stages = [
            (np.array([0.3, 0.2]),
             np.array([[1.6, 0.50],[0.40, 0.20],[0.20, 0.10],[0.10, 0.05]]),
             np.array([[0.30, 0.80],[0.10, 0.20],[0.05, 0.10],[0.02, 0.05]]),
             "Gen 1 — aleatoria", "#FF4444"),
            (np.array([0.1, 0.30]),
             np.array([[1.30, 0.10],[0.10, -0.15],[0.05, 0.00],[0.02, -0.02]]),
             np.array([[0.10, 0.65],[0.05, 0.03],[0.02, 0.00],[0.01, 0.00]]),
             "Gen 40 — mejorando", "#FF9900"),
            (np.array([0.04, 0.34]),
             np.array([[1.20, 0.00],[0.00, -0.23],[0.00, 0.00],[0.00, -0.04]]),
             np.array([[0.00, 0.61],[0.00, 0.02],[0.00, 0.00],[0.00, 0.00]]),
             "Gen 100 — convergiendo", "#FFDD00"),
            (C0, A0, B0, "Gen 150 — convergido ✓", C_C0),
        ]

        gen_lbl = Text("", font_size=26, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.add(gen_lbl)
        cur = None
        for c, a, b, txt, col in stages:
            mob = make_curve(ax, c, a, b, col, stroke=3)
            new_lbl = Text(txt, font_size=26, color=col).to_edge(DOWN, buff=0.3)
            if cur is None:
                self.play(Create(mob), ReplacementTransform(gen_lbl, new_lbl), run_time=1.2)
            else:
                self.play(Transform(cur, mob), ReplacementTransform(gen_lbl, new_lbl), run_time=1.4)
            gen_lbl = new_lbl
            cur = mob
            self.wait(0.9)

        self.wait(1.5)
        self.play(FadeOut(VGroup(hdr, ax, dots, cur, gen_lbl)))

    # ── 8. Manifold final ─────────────────────────────────────────────────────
    def _final_manifold(self):
        hdr = self._hdr("Paso 7 — Manifolds Convergidos (One-vs-Rest, 2 clases)")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, d1 = two_moons(n=40)
        dots0 = VGroup(*[Dot(ax.c2p(x,y), radius=0.08, color=C_C0) for x,y in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y), radius=0.08, color=C_C1) for x,y in d1])
        cv0 = make_curve(ax, C0, A0, B0, C_C0, stroke=3)
        cv1 = make_curve(ax, C1, A1, B1, C_C1, stroke=3)
        ct0 = Dot(ax.c2p(*C0), radius=0.11, color=C_C0, fill_opacity=0.9)
        ct1 = Dot(ax.c2p(*C1), radius=0.11, color=C_C1, fill_opacity=0.9)
        cl0 = MathTex(r"\mathbf{c}_0", font_size=22, color=C_C0).next_to(ct0, UL, buff=0.05)
        cl1 = MathTex(r"\mathbf{c}_1", font_size=22, color=C_C1).next_to(ct1, UR, buff=0.05)


        self.play(Create(ax))
        self.play(FadeIn(dots0, dots1))
        self.play(Create(cv0), GrowFromCenter(ct0), Write(cl0), run_time=1.2)
        self.play(Create(cv1), GrowFromCenter(ct1), Write(cl1), run_time=1.2)

        cap = Text("Cada clase tiene su propio manifold Mₖ con centroide local cₖ",
                   font_size=22, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(VGroup(hdr, ax, dots0, dots1, cv0, cv1, ct0, ct1, cl0, cl1, cap)))

    # ── 9. Inferencia ─────────────────────────────────────────────────────────
    def _inference(self):
        self.clear()
        hdr = self._hdr("Paso 8 — Inferencia: Nuevo Punto x*")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, d1 = two_moons(n=40)
        dots0 = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C0, fill_opacity=0.35) for x,y in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C1, fill_opacity=0.35) for x,y in d1])
        cv0 = make_curve(ax, C0, A0, B0, C_C0, stroke=2.5)
        cv1 = make_curve(ax, C1, A1, B1, C_C1, stroke=2.5)
        ct0 = Dot(ax.c2p(*C0), radius=0.1, color=C_C0, fill_opacity=0.9)
        ct1 = Dot(ax.c2p(*C1), radius=0.1, color=C_C1, fill_opacity=0.9)
        self.play(Create(ax), FadeIn(dots0, dots1), Create(cv0), Create(cv1),
                  GrowFromCenter(ct0), GrowFromCenter(ct1), run_time=1.5)

        # Nuevo punto
        xp = np.array([0.15, 0.52])
        ndot = Dot(ax.c2p(*xp), radius=0.13, color=C_NEW)
        ndot.set_style(stroke_color=WHITE, stroke_width=2)
        qm = Text("?", font_size=28, color=C_NEW, weight=BOLD).next_to(ndot, UR, buff=0.05)
        self.play(Flash(ax.c2p(*xp), color=C_NEW, flash_radius=0.5))
        self.play(GrowFromCenter(ndot), Write(qm))
        self.wait(0.5)

        # Proyección
        ds0, g0 = radial_ds(xp, C0, A0, B0)
        ds1, g1 = radial_ds(xp, C1, A1, B1)

        pd0 = Dot(ax.c2p(*g0), radius=0.09, color=C_C0)
        pd1 = Dot(ax.c2p(*g1), radius=0.09, color=C_C1)
        ld0 = DashedLine(ax.c2p(*xp), ax.c2p(*g0), color=C_C0, stroke_width=2, dash_length=0.1)
        ld1 = DashedLine(ax.c2p(*xp), ax.c2p(*g1), color=C_C1, stroke_width=2, dash_length=0.1)

        step1 = Text("1. Proyección t* — punto más cercano γ(t*) en cada curva",
                     font_size=20, color=C_TEXT).to_edge(DOWN, buff=0.45)
        self.play(Write(step1))
        self.play(Create(ld0), GrowFromCenter(pd0),
                  Create(ld1), GrowFromCenter(pd1), run_time=1.2)
        self.wait(0.8)

        # Radios
        rx0 = Line(ax.c2p(*xp), ax.c2p(*C0), color=C_C0, stroke_width=2.5)
        rg0 = Line(ax.c2p(*g0), ax.c2p(*C0), color=C_C0, stroke_width=2, stroke_opacity=0.6)
        rx1 = Line(ax.c2p(*xp), ax.c2p(*C1), color=C_C1, stroke_width=2.5)
        rg1 = Line(ax.c2p(*g1), ax.c2p(*C1), color=C_C1, stroke_width=2, stroke_opacity=0.6)

        step2 = Text("2. Comparación radial: DS = d(x,c) − d(γ(t*),c)",
                     font_size=20, color=C_TEXT).to_edge(DOWN, buff=0.45)
        self.play(ReplacementTransform(step1, step2))
        self.play(Create(rx0), Create(rg0), Create(rx1), Create(rg1), run_time=1.2)
        self.wait(0.8)

        # Fórmulas numéricas
        fm = VGroup(
            MathTex(rf"D_S^{{(0)}} = {ds0:.3f}", font_size=26, color=C_C0),
            MathTex(rf"D_S^{{(1)}} = {ds1:.3f}", font_size=26, color=C_C1),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR, buff=0.5)

        self.play(FadeIn(fm))
        self.wait(0.8)

        # Decisión
        winner = 0 if ds0 < ds1 else 1
        wcol   = C_C0 if winner == 0 else C_C1
        dec = Text(f"3.  argmin DS → Clase {winner}  (más negativo = más dentro)",
                   font_size=24, color=wcol, weight=BOLD).to_edge(DOWN, buff=0.45)
        self.play(ReplacementTransform(step2, dec))
        self.play(qm.animate.become(
            Text(str(winner), font_size=28, color=wcol, weight=BOLD).next_to(ndot, UR, buff=0.05)
        ))
        self.play(Flash(ax.c2p(*xp), color=wcol, flash_radius=0.6))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))

    # ── 9.5 Escena 3D ─────────────────────────────────────────────────────────
    def _scene_3d(self):
        self.clear()
        hdr = Text("Escalabilidad: 3 Clases en 3 Dimensiones (One-vs-Rest)", font_size=30, color=C_ACC, weight=BOLD).to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(hdr)
        self.play(FadeIn(hdr))
        
        ax = ThreeDAxes(
            x_range=[-2.5, 2.5, 1], y_range=[-1.5, 2.5, 1], z_range=[-2.5, 2.5, 1],
            x_length=8, y_length=8, z_length=8,
            axis_config={"color": C_DIM}
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(ax))
        
        d0, d1, d2 = blobs_3d(n=40)
        # Dot es 2D pero se dibuja correctamente en coordenadas 3D al transformarlo, rinde muchísimo mejor
        dots0 = VGroup(*[Dot(ax.c2p(x,y,z), radius=0.08, color=C_C0) for x,y,z in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y,z), radius=0.08, color=C_C1) for x,y,z in d1])
        dots2 = VGroup(*[Dot(ax.c2p(x,y,z), radius=0.08, color=C_C2) for x,y,z in d2])
        
        self.play(FadeIn(dots0, dots1, dots2))
        
        cv0 = make_curve_3d(ax, C0_3, A0_3, B0_3, C_C0, stroke=4)
        cv1 = make_curve_3d(ax, C1_3, A1_3, B1_3, C_C1, stroke=4)
        cv2 = make_curve_3d(ax, C2_3, A2_3, B2_3, C_C2, stroke=4)
        
        ct0 = Dot(ax.c2p(*C0_3), radius=0.12, color=C_C0)
        ct1 = Dot(ax.c2p(*C1_3), radius=0.12, color=C_C1)
        ct2 = Dot(ax.c2p(*C2_3), radius=0.12, color=C_C2)
        
        self.play(Create(cv0), GrowFromCenter(ct0))
        self.play(Create(cv1), GrowFromCenter(ct1))
        self.play(Create(cv2), GrowFromCenter(ct2))
        
        # Rotar cámara para ver 3D
        self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=4)
        self.move_camera(phi=80 * DEGREES, theta=120 * DEGREES, run_time=5)
        
        cap = Text("Un manifold independiente (curva de Fourier 1D) para cada clase en ℝ³", font_size=24, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(Write(cap))
        self.wait(3)
        
        self.play(FadeOut(VGroup(hdr, ax, dots0, dots1, dots2, cv0, cv1, cv2, ct0, ct1, ct2, cap)))
        
        # Restaurar la cámara a la proyección 2D estándar para la siguiente escena
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)


    # ── 10. Outro ─────────────────────────────────────────────────────────────
    def _outro(self):
        items = VGroup(
            Text("✓  Curva 1D de Fourier en ℝᴰ — frontera continua y diferenciable", font_size=24, color=C_C0),
            Text("✓  NSGA-II bi-objetivo: error  +  Navaja de Ockham", font_size=24, color=C_C0),
            Text("✓  Signo por comparación radial al centroide (star-shaped)", font_size=24, color=C_C0),
            Text("✓  Pesos w co-evolucionados — feature selection implícito", font_size=24, color=C_ACC),
            Text("✓  Paralelismo lock-free multi-núcleo (OpenMP)", font_size=24, color=C_C0),
            Text("✓  Ecuación paramétrica explícita → 100 % White-Box", font_size=24, color=C_PARETO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).center()

        title = Text("FM-NSGA-II — Resumen del Algoritmo",
                     font_size=38, weight=BOLD, color=C_TEXT).next_to(items, UP, buff=0.5)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[Write(i) for i in items], lag_ratio=0.22), run_time=3)
        self.wait(3)
        self.play(FadeOut(VGroup(title, items)))

