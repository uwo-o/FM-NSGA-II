"""
AM_NSGA_II_Animation.py — Manim Community Edition
===================================================
Renderizar con:
    manim -pqh AM_NSGA_II_Animation.py AMNSGAFullAnimation

Escenas:
  1. Portada
  2. Dataset Two-Moons
  3. Cromosoma Angular (centro, vector principal v, coeficientes Chebyshev)
  4. Población inicial aleatoria
  5. Evaluación de fitness (O1, O2)
  6. Frente de Pareto / selección
  7. Evolución generacional (morphing del radio angular)
  8. Manifolds convergidos con vector v
  9. Inferencia: distancia angular + clase
 10. Escena 3D (esferas angulares deformadas)
 11. Resumen / Outro
"""

from manim import *
import numpy as np

# ─── Paleta ───────────────────────────────────────────────────────────────────
C_BG     = "#0D1117"
C_C0     = "#00C6FF"   # azul cian — clase 0
C_C1     = "#FF6B9D"   # rosa — clase 1
C_PARETO = "#F7C59F"
C_NEW    = "#FFD166"
C_TEXT   = "#E8E8E8"
C_DIM    = "#555D6B"
C_ACC    = "#A259FF"   # violeta — acento principal
C_OBJ1   = "#06D6A0"
C_OBJ2   = "#EF476F"
C_VEC    = "#FFB703"   # dorado — vector principal v
C_C2     = "#FFE66D"

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

# ─── Angular Manifold helpers ─────────────────────────────────────────────────
def chebyshev_radius(cos_theta, coefs):
    """R(θ) = a_0 + Σ_{k=1}^H a_k · T_k(cos θ)"""
    r = coefs[0]
    if len(coefs) < 2:
        return r
    t_prev = 1.0
    t_curr = cos_theta
    r += coefs[1] * t_curr
    for k in range(2, len(coefs)):
        t_next = 2.0 * cos_theta * t_curr - t_prev
        r += coefs[k] * t_next
        t_prev = t_curr
        t_curr = t_next
    return r

def angular_boundary_xy(center, v, coefs, n=300):
    """Frontera del AngularManifold en 2D."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = center
    vx, vy = v
    vnorm = np.sqrt(vx**2 + vy**2)
    if vnorm < 1e-10:
        vx, vy, vnorm = 1.0, 0.0, 1.0
    vx /= vnorm; vy /= vnorm
    pts = []
    for p in phi:
        dx, dy = np.cos(p), np.sin(p)
        cos_theta = np.clip(dx * vx + dy * vy, -1.0, 1.0)
        R = max(chebyshev_radius(cos_theta, coefs), 0.01)
        pts.append([cx + R * dx, cy + R * dy])
    return np.array(pts)

def make_angular_curve(ax, center, v, coefs, color, stroke=3):
    pts = angular_boundary_xy(center, v, coefs)
    p3d = [ax.c2p(p[0], p[1]) for p in pts]
    m = VMobject(color=color, stroke_width=stroke)
    m.set_points_as_corners(p3d + [p3d[0]])
    return m

def angular_signed_distance(pt, center, v, coefs):
    """d(x) = ||x - c|| - R(cos θ)"""
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    r = np.sqrt(dx**2 + dy**2)
    vx, vy = v
    vnorm = np.sqrt(vx**2 + vy**2)
    if vnorm < 1e-10 or r < 1e-10:
        return r - coefs[0], np.array(center)
    vx /= vnorm; vy /= vnorm
    cos_theta = np.clip((dx * vx + dy * vy) / r, -1.0, 1.0)
    R_theta = max(chebyshev_radius(cos_theta, coefs), 0.01)
    ux, uy = dx / r, dy / r
    gamma = np.array([center[0] + R_theta * ux, center[1] + R_theta * uy])
    return r - R_theta, gamma

# ─── Manifolds analíticos para Two-Moons ─────────────────────────────────────
C0     = np.array([0.0, 0.32])
V0     = np.array([0.0, 1.0])
COEFS0 = np.array([0.82, 0.22, -0.12, 0.05])

C1     = np.array([1.0, 0.18])
V1     = np.array([0.0, -1.0])
COEFS1 = np.array([0.82, 0.22, -0.12, 0.05])

# Curvas aleatorias iniciales
rng0 = np.random.default_rng(7)
RAND_CURVES = []
for _ in range(6):
    c_rand    = rng0.uniform([-0.2, -0.2], [1.2, 0.8])
    v_rand    = rng0.uniform(-1.0, 1.0, 2)
    coefs_r   = np.concatenate([[rng0.uniform(0.2, 1.0)],
                                 rng0.uniform(-0.5, 0.5, 2)])
    RAND_CURVES.append((c_rand, v_rand, coefs_r))

# ─── Datos 3D ─────────────────────────────────────────────────────────────────
def blobs_3d(n=40, seed=42):
    rng = np.random.default_rng(seed)
    c0 = rng.normal([0, 1.5, 0],    0.35, (n, 3))
    c1 = rng.normal([1.5, -0.5, 1.5],  0.35, (n, 3))
    c2 = rng.normal([-1.5, -0.5, -1.5], 0.35, (n, 3))
    return c0, c1, c2

def angular_sphere_pts(center, v, coefs, n_phi=20, n_theta=20):
    phi   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta = np.linspace(0, np.pi, n_theta)
    cx, cy, cz = center
    vx, vy, vz = v
    vnorm = np.sqrt(vx**2 + vy**2 + vz**2)
    if vnorm < 1e-10: vx, vy, vz, vnorm = 0, 0, 1, 1.0
    vx /= vnorm; vy /= vnorm; vz /= vnorm
    pts = []
    for t in theta:
        for p in phi:
            dx = np.sin(t)*np.cos(p)
            dy = np.sin(t)*np.sin(p)
            dz = np.cos(t)
            cos_a = np.clip(dx*vx + dy*vy + dz*vz, -1.0, 1.0)
            R = max(chebyshev_radius(cos_a, coefs), 0.01)
            pts.append([cx + R*dx, cy + R*dy, cz + R*dz])
    return np.array(pts)

def make_sphere_surface(ax, center, v, coefs, color, n=16):
    pts = angular_sphere_pts(center, v, coefs, n_phi=n, n_theta=n)
    return VGroup(*[Dot(ax.c2p(p[0], p[1], p[2]),
                        radius=0.04, color=color, fill_opacity=0.5)
                    for p in pts])

C0_3 = np.array([0, 1.5, 0]);    V0_3 = np.array([0, 1, 0]);   COEFS0_3 = np.array([1.1, 0.25, -0.1])
C1_3 = np.array([1.5, -0.5, 1.5]); V1_3 = np.array([1, 0, 0]); COEFS1_3 = np.array([1.1, 0.25, -0.1])
C2_3 = np.array([-1.5, -0.5, -1.5]); V2_3 = np.array([0, 0, -1]); COEFS2_3 = np.array([1.1, 0.25, -0.1])


# ══════════════════════════════════════════════════════════════════════════════
class AMNSGAFullAnimation(ThreeDScene):

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
            Text("AM-NSGA-II", font_size=76, weight=BOLD, color=C_C0),
            Text("Angular Manifold · Non-Dominated Sorting GA", font_size=28, color=C_TEXT),
            Text("Kernel Topológico Polar con Polinomios de Chebyshev", font_size=24, color=C_DIM),
        ).arrange(DOWN, buff=0.45).center()

        subtitle_box = SurroundingRectangle(
            title[2], color=C_ACC, buff=0.15, corner_radius=0.1,
            stroke_width=1.5, fill_opacity=0.05, fill_color=C_ACC
        )

        self.play(Write(title[0]), run_time=1.5)
        self.play(FadeIn(title[1], title[2], subtitle_box), run_time=1)
        self.wait(2)
        self.play(FadeOut(VGroup(title, subtitle_box)))

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
        q = Text("¿Cómo separar estas lunas con una frontera angular-radial?",
                 font_size=24, color=C_PARETO).to_edge(DOWN, buff=0.3)
        self.play(Write(q))
        self.wait(2)
        self.play(FadeOut(VGroup(hdr, ax, dots0, dots1, q)))

    # ── 3. Cromosoma Angular ──────────────────────────────────────────────────
    def _chromosome(self):
        hdr = self._hdr("Paso 2 — El Kernel Topológico Angular")
        self.play(FadeIn(hdr))

        eq = MathTex(
            r"d(\mathbf{x}) = \underbrace{\|\mathbf{x} - \mathbf{c}\|_w}_{r}"
            r" - \underbrace{R(\cos\theta)}_{\text{radio angular}}",
            font_size=36, color=C_TEXT
        ).shift(UP * 2.7)
        self.play(Write(eq), run_time=2)

        eq2 = MathTex(
            r"R(\cos\theta) = a_0 + \sum_{k=1}^{H} a_k\, T_k(\cos\theta)",
            font_size=30, color=C_ACC
        ).shift(UP * 1.6)
        self.play(Write(eq2), run_time=1.5)

        # Visualización geométrica del manifold angular
        mini_ax = Axes(
            x_range=[-1.6, 1.6, 0.5], y_range=[-1.6, 1.6, 0.5],
            x_length=3.5, y_length=3.5,
            axis_config={"color": C_DIM, "include_tip": False},
        ).shift(DOWN * 0.8 + LEFT * 3.2)
        demo_curve  = make_angular_curve(mini_ax, [0.0, 0.0], [0.0, 1.0],
                                          [0.9, 0.35, -0.15], C_C0, stroke=2.5)
        demo_center = Dot(mini_ax.c2p(0, 0), radius=0.1, color=C_VEC)

        v_arrow = Arrow(
            mini_ax.c2p(0, 0), mini_ax.c2p(0, 0.85),
            color=C_VEC, buff=0, stroke_width=3, max_tip_length_to_length_ratio=0.15
        )
        v_lbl = MathTex(r"\mathbf{v}", font_size=22, color=C_VEC).next_to(v_arrow.get_end(), UP, buff=0.05)

        arc = Arc(radius=0.45, start_angle=PI/2, angle=-PI/3,
                  color=C_PARETO, stroke_width=2).move_arc_center_to(mini_ax.c2p(0, 0))
        theta_lbl = MathTex(r"\theta", font_size=20, color=C_PARETO).move_to(mini_ax.c2p(0.28, 0.40))

        self.play(Create(mini_ax), Create(demo_curve), GrowFromCenter(demo_center))
        self.play(GrowArrow(v_arrow), Write(v_lbl))
        self.play(Create(arc), Write(theta_lbl))

        # Tabla de genes del cromosoma
        specs = [
            (r"c_x",    "#E63946"),
            (r"c_y",    "#E63946"),
            (r"v_x",    C_VEC),
            (r"v_y",    C_VEC),
            (r"w_1",    C_ACC),
            (r"a_0",    "#2A9D8F"),
            (r"a_1",    "#2A9D8F"),
            (r"\cdots", C_DIM),
            (r"a_H",    "#2A9D8F"),
        ]
        row = VGroup()
        for lbl, col in specs:
            box = Rectangle(width=0.82, height=0.56,
                            fill_opacity=0.3, fill_color=col,
                            stroke_color=col, stroke_width=2)
            lab = MathTex(lbl, font_size=20, color=WHITE)
            row.add(VGroup(box, lab))
        row.arrange(RIGHT, buff=0.04).shift(DOWN * 2.4 + RIGHT * 1.2)

        lbl_row = Text("Cromosoma", font_size=17, color=C_TEXT).next_to(row, LEFT, buff=0.15)
        brace_c = Brace(VGroup(*list(row)[:2]), DOWN, buff=0.05, color="#E63946")
        brace_v = Brace(VGroup(*list(row)[2:4]), DOWN, buff=0.05, color=C_VEC)
        brace_w = Brace(VGroup(*list(row)[4:5]), DOWN, buff=0.05, color=C_ACC)
        brace_a = Brace(VGroup(*list(row)[5:]), DOWN, buff=0.05, color="#2A9D8F")
        bt_c = Tex("centro",    font_size=14).set_color("#E63946"); brace_c.put_at_tip(bt_c)
        bt_v = Tex("eje polar", font_size=14).set_color(C_VEC);    brace_v.put_at_tip(bt_v)
        bt_w = Tex("pesos",     font_size=14).set_color(C_ACC);    brace_w.put_at_tip(bt_w)
        bt_a = Tex("Chebyshev", font_size=14).set_color("#2A9D8F"); brace_a.put_at_tip(bt_a)

        self.play(LaggedStart(*[GrowFromCenter(c) for c in row], lag_ratio=0.07))
        self.play(Write(lbl_row))
        self.play(FadeIn(brace_c, bt_c, brace_v, bt_v, brace_w, bt_w, brace_a, bt_a))

        info = Text(
            "v co-evoluciona: orienta la asimetría  |  a_k modula el radio según cada dirección θ",
            font_size=17, color=C_DIM
        ).to_edge(DOWN, buff=0.2)
        self.play(Write(info))
        self.wait(2.5)
        self.play(FadeOut(VGroup(hdr, eq, eq2, mini_ax, demo_curve, demo_center,
                                 v_arrow, v_lbl, arc, theta_lbl,
                                 row, lbl_row, brace_c, bt_c, brace_v, bt_v,
                                 brace_w, bt_w, brace_a, bt_a, info)))

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
        for i, (c, v, coefs) in enumerate(RAND_CURVES):
            m = make_angular_curve(ax, c, v, coefs, cols[i], stroke=2)
            m.set_opacity(0.65)
            mobs.add(m)

        self.play(LaggedStart(*[Create(m) for m in mobs], lag_ratio=0.2), run_time=2.5)
        bad = Text("Fitness alto — fronteras angulares mal orientadas (O₁ ≈ 0.5)",
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
        curve = make_angular_curve(ax, C0, V0, COEFS0, C_C0, stroke=3)

        self.play(Create(ax), FadeIn(dots), Create(curve))

        o1 = VGroup(
            Text("O₁ — Error de clasificación", font_size=21, color=C_OBJ1, weight=BOLD),
            MathTex(r"O_1 = \frac{1}{N}\sum_i \mathbb{I}[y_i \neq \hat y_i]",
                    font_size=26, color=C_TEXT),
            Text("→ minimizar", font_size=18, color=C_OBJ1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        o2 = VGroup(
            Text("O₂ — Complejidad topológica", font_size=21, color=C_OBJ2, weight=BOLD),
            MathTex(r"O_2 = \lambda\frac{\sum|a_k|}{E_0}+(1-\lambda)\frac{H}{H_{\max}}",
                    font_size=26, color=C_TEXT),
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
        yl = Text("O₂ (complejidad angular)", font_size=22, color=C_OBJ2
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
        hdr = self._hdr("Paso 6 — Evolución Generacional del Kernel Angular")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, _ = two_moons(n=40)
        dots = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C0, fill_opacity=0.6) for x,y in d0])
        self.play(Create(ax), FadeIn(dots))

        stages = [
            (np.array([0.4, 0.0]),  np.array([1.0, 0.5]),
             np.array([1.1, 0.0, 0.0]),
             "Gen 1 — forma circular aleatoria", "#FF4444"),
            (np.array([0.2, 0.15]), np.array([0.2, 1.0]),
             np.array([0.95, 0.30, -0.05]),
             "Gen 40 — orientándose hacia la luna", "#FF9900"),
            (np.array([0.05, 0.28]), np.array([0.0, 1.0]),
             np.array([0.88, 0.28, -0.10, 0.03]),
             "Gen 100 — convergiendo asimetría", "#FFDD00"),
            (C0, V0, COEFS0, "Gen 150 — kernel angular convergido ✓", C_C0),
        ]

        gen_lbl = Text("", font_size=26, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.add(gen_lbl)
        cur = None
        for c, v, coefs, txt, col in stages:
            mob = make_angular_curve(ax, c, v, coefs, col, stroke=3)
            new_lbl = Text(txt, font_size=24, color=col).to_edge(DOWN, buff=0.3)
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
        hdr = self._hdr("Paso 7 — Manifolds Angulares Convergidos (One-vs-Rest)")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, d1 = two_moons(n=40)
        dots0 = VGroup(*[Dot(ax.c2p(x,y), radius=0.08, color=C_C0) for x,y in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y), radius=0.08, color=C_C1) for x,y in d1])
        cv0 = make_angular_curve(ax, C0, V0, COEFS0, C_C0, stroke=3)
        cv1 = make_angular_curve(ax, C1, V1, COEFS1, C_C1, stroke=3)
        ct0 = Dot(ax.c2p(*C0), radius=0.11, color=C_C0, fill_opacity=0.9)
        ct1 = Dot(ax.c2p(*C1), radius=0.11, color=C_C1, fill_opacity=0.9)

        # Vectores principales
        v0_end = ax.c2p(C0[0], C0[1] + 0.6)
        v1_end = ax.c2p(C1[0], C1[1] - 0.6)
        v0_arr = Arrow(ax.c2p(*C0), v0_end, color=C_VEC, buff=0,
                       stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        v1_arr = Arrow(ax.c2p(*C1), v1_end, color=C_VEC, buff=0,
                       stroke_width=2.5, max_tip_length_to_length_ratio=0.18)

        cl0 = MathTex(r"\mathbf{c}_0", font_size=22, color=C_C0).next_to(ct0, UL, buff=0.05)
        cl1 = MathTex(r"\mathbf{c}_1", font_size=22, color=C_C1).next_to(ct1, UR, buff=0.05)
        vl0 = MathTex(r"\mathbf{v}_0", font_size=20, color=C_VEC).next_to(v0_arr.get_end(), RIGHT, buff=0.05)
        vl1 = MathTex(r"\mathbf{v}_1", font_size=20, color=C_VEC).next_to(v1_arr.get_end(), RIGHT, buff=0.05)

        self.play(Create(ax))
        self.play(FadeIn(dots0, dots1))
        self.play(Create(cv0), GrowFromCenter(ct0), Write(cl0), GrowArrow(v0_arr), Write(vl0), run_time=1.2)
        self.play(Create(cv1), GrowFromCenter(ct1), Write(cl1), GrowArrow(v1_arr), Write(vl1), run_time=1.2)

        cap = Text("El vector v define el eje polar: R(θ) varía el radio según la dirección angular",
                   font_size=20, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(VGroup(hdr, ax, dots0, dots1, cv0, cv1, ct0, ct1,
                                 cl0, cl1, vl0, vl1, v0_arr, v1_arr, cap)))

    # ── 9. Inferencia ─────────────────────────────────────────────────────────
    def _inference(self):
        self.clear()
        hdr = self._hdr("Paso 8 — Inferencia: Nuevo Punto x*")
        self.play(FadeIn(hdr))
        ax = self._std_axes()
        d0, d1 = two_moons(n=40)
        dots0 = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C0, fill_opacity=0.35) for x,y in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y), radius=0.07, color=C_C1, fill_opacity=0.35) for x,y in d1])
        cv0 = make_angular_curve(ax, C0, V0, COEFS0, C_C0, stroke=2.5)
        cv1 = make_angular_curve(ax, C1, V1, COEFS1, C_C1, stroke=2.5)
        ct0 = Dot(ax.c2p(*C0), radius=0.1, color=C_C0, fill_opacity=0.9)
        ct1 = Dot(ax.c2p(*C1), radius=0.1, color=C_C1, fill_opacity=0.9)
        self.play(Create(ax), FadeIn(dots0, dots1), Create(cv0), Create(cv1),
                  GrowFromCenter(ct0), GrowFromCenter(ct1), run_time=1.5)

        xp = np.array([0.15, 0.52])
        ndot = Dot(ax.c2p(*xp), radius=0.13, color=C_NEW)
        ndot.set_style(stroke_color=WHITE, stroke_width=2)
        qm = Text("?", font_size=28, color=C_NEW, weight=BOLD).next_to(ndot, UR, buff=0.05)
        self.play(Flash(ax.c2p(*xp), color=C_NEW, flash_radius=0.5))
        self.play(GrowFromCenter(ndot), Write(qm))
        self.wait(0.5)

        ds0, g0 = angular_signed_distance(xp, C0, V0, COEFS0)
        ds1, g1 = angular_signed_distance(xp, C1, V1, COEFS1)

        pd0 = Dot(ax.c2p(*g0), radius=0.09, color=C_C0)
        pd1 = Dot(ax.c2p(*g1), radius=0.09, color=C_C1)
        ld0 = DashedLine(ax.c2p(*xp), ax.c2p(*g0), color=C_C0, stroke_width=2, dash_length=0.1)
        ld1 = DashedLine(ax.c2p(*xp), ax.c2p(*g1), color=C_C1, stroke_width=2, dash_length=0.1)

        step1 = Text("1. Calcular θ = ángulo entre (x−c) y v  →  R(cos θ) vía Chebyshev",
                     font_size=19, color=C_TEXT).to_edge(DOWN, buff=0.45)
        self.play(Write(step1))
        self.play(Create(ld0), GrowFromCenter(pd0),
                  Create(ld1), GrowFromCenter(pd1), run_time=1.2)
        self.wait(0.8)

        rx0 = Line(ax.c2p(*xp), ax.c2p(*C0), color=C_C0, stroke_width=2.5)
        rx1 = Line(ax.c2p(*xp), ax.c2p(*C1), color=C_C1, stroke_width=2.5)

        step2 = Text("2. d(x) = ‖x − c‖ − R(cos θ)   [negativo = dentro]",
                     font_size=19, color=C_TEXT).to_edge(DOWN, buff=0.45)
        self.play(ReplacementTransform(step1, step2))
        self.play(Create(rx0), Create(rx1), run_time=1.2)
        self.wait(0.8)

        fm = VGroup(
            MathTex(rf"d^{{(0)}} = {ds0:.3f}", font_size=26, color=C_C0),
            MathTex(rf"d^{{(1)}} = {ds1:.3f}", font_size=26, color=C_C1),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR, buff=0.5)
        self.play(FadeIn(fm))
        self.wait(0.8)

        winner = 0 if ds0 < ds1 else 1
        wcol   = C_C0 if winner == 0 else C_C1
        dec = Text(f"3.  argmin d → Clase {winner}  (más negativo = más dentro del manifold)",
                   font_size=22, color=wcol, weight=BOLD).to_edge(DOWN, buff=0.45)
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
        hdr = Text("Escalabilidad: 3 Clases en 3D — Esferas Angulares Deformadas",
                   font_size=28, color=C_ACC, weight=BOLD).to_edge(UP, buff=0.2)
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
        dots0 = VGroup(*[Dot(ax.c2p(x,y,z), radius=0.08, color=C_C0) for x,y,z in d0])
        dots1 = VGroup(*[Dot(ax.c2p(x,y,z), radius=0.08, color=C_C1) for x,y,z in d1])
        dots2 = VGroup(*[Dot(ax.c2p(x,y,z), radius=0.08, color=C_C2) for x,y,z in d2])
        self.play(FadeIn(dots0, dots1, dots2))

        sph0 = make_sphere_surface(ax, C0_3, V0_3, COEFS0_3, C_C0, n=16)
        sph1 = make_sphere_surface(ax, C1_3, V1_3, COEFS1_3, C_C1, n=16)
        sph2 = make_sphere_surface(ax, C2_3, V2_3, COEFS2_3, C_C2, n=16)

        ct0 = Dot(ax.c2p(*C0_3), radius=0.12, color=C_C0)
        ct1 = Dot(ax.c2p(*C1_3), radius=0.12, color=C_C1)
        ct2 = Dot(ax.c2p(*C2_3), radius=0.12, color=C_C2)

        self.play(FadeIn(sph0), GrowFromCenter(ct0))
        self.play(FadeIn(sph1), GrowFromCenter(ct1))
        self.play(FadeIn(sph2), GrowFromCenter(ct2))

        self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=4)
        self.move_camera(phi=80 * DEGREES, theta=120 * DEGREES, run_time=5)

        cap = Text("AngularManifold: hipersuperficie radial en ℝ³ — R(θ) varía con Chebyshev",
                   font_size=21, color=C_TEXT).to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(cap)
        self.play(Write(cap))
        self.wait(3)
        self.play(FadeOut(VGroup(hdr, ax, dots0, dots1, dots2,
                                 sph0, sph1, sph2, ct0, ct1, ct2, cap)))
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

    # ── 10. Outro ─────────────────────────────────────────────────────────────
    def _outro(self):
        items = VGroup(
            Text("✓  Hipersuperficie angular R(θ)  — frontera radial adaptativa", font_size=24, color=C_C0),
            Text("✓  Polinomios de Chebyshev T_k — base ortogonal, bajo costo O(H)", font_size=24, color=C_C0),
            Text("✓  NSGA-II bi-objetivo: error  +  Navaja de Ockham (L1 + H)", font_size=24, color=C_C0),
            Text("✓  Signo por comparación radial: d(x) = r − R(cos θ)", font_size=24, color=C_ACC),
            Text("✓  Vector v co-evolucionado — orienta la asimetría de la frontera", font_size=24, color=C_VEC),
            Text("✓  Ecuación analítica cerrada → 100 % White-Box & diferenciable", font_size=24, color=C_PARETO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).center()

        title = Text("AM-NSGA-II — Kernel Topológico Angular",
                     font_size=38, weight=BOLD, color=C_TEXT).next_to(items, UP, buff=0.5)

        self.play(FadeIn(title))
        self.play(LaggedStart(*[Write(i) for i in items], lag_ratio=0.22), run_time=3)
        self.wait(3)
        self.play(FadeOut(VGroup(title, items)))
