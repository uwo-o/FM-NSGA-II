#pragma once
// ============================================================
// include/mgda.hpp — Multiple Gradient Descent Algorithm (MGDA)
// ============================================================
//
// Implementa descenso de gradiente multi-objetivo para AngularManifold.
// En cada iteración:
//   1. Calcula g1 = ∇O1 (gradiente del Error / Logistic Loss)
//   2. Calcula g2 = ∇O2 (gradiente de la Complejidad)
//   3. Resuelve el QP mínimo-norma: find α ∈ [0,1] s.t. ||α·g1+(1-α)·g2|| es mínimo
//   4. Aplica la dirección combinada d = α·g1 + (1-α)·g2
//   5. Si ||d|| < tol, aterriza en un punto del frente de Pareto → STOP
//
// Referencias:
//   Désidéri (2012). "Multiple-gradient descent algorithm (MGDA)
//   for multiobjective optimization." C. R. Math. 350(5–6), pp. 313–318.
// ============================================================
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include "angular_manifold.hpp"
#include "dataset.hpp"

namespace mgda {

// ─── Configuración ────────────────────────────────────────────
struct MGDAConfig {
    double lr           = 5e-3;   // learning rate inicial
    double lr_decay     = 0.999;  // decaimiento multiplicativo por iteración
    int    n_iter       = 3000;   // máx. iteraciones
    int    batch_size   = 128;    // tamaño de mini-batch (0 = full batch)
    double tol          = 1e-6;   // convergencia: ||d|| < tol → Pareto point
    double lambda_l1    = 0.1;    // peso L1 de los feature weights en O2
    int    n_harmonics  = 3;      // armónicos iniciales del manifold
    bool   verbose      = true;
    int    log_every    = 200;
};

// ─── Punto de trayectoria (para exportar y visualizar) ────────
struct TrajectoryPoint {
    int    iter;
    double error;       // O1
    double complexity;  // O2
    double alpha;       // coeficiente de mezcla MGDA
    double grad_norm;   // ||d|| = ||alpha*g1 + (1-alpha)*g2||
};

// ─── Buffer de gradiente (mismo layout que el cromosoma) ──────
// Layout: center (dim), v (dim), w (dim), coefs (H+1)
struct GradBuf {
    int dim, H;
    std::vector<double> center, v, w, coefs;

    GradBuf(int d, int h)
        : dim(d), H(h), center(d,0.), v(d,0.), w(d,0.), coefs(h+1,0.) {}

    void zero() {
        std::fill(center.begin(), center.end(), 0.);
        std::fill(v.begin(),      v.end(),      0.);
        std::fill(w.begin(),      w.end(),      0.);
        std::fill(coefs.begin(),  coefs.end(),  0.);
    }

    // Producto punto en el espacio de parámetros completo
    double dot(const GradBuf& o) const {
        double s = 0.;
        for (int i=0;i<dim;++i) s += center[i]*o.center[i];
        for (int i=0;i<dim;++i) s += v[i]*o.v[i];
        for (int i=0;i<dim;++i) s += w[i]*o.w[i];
        for (int k=0;k<=H;++k) s += coefs[k]*o.coefs[k];
        return s;
    }
    double norm_sq() const { return dot(*this); }

    // Combinación convexa: alpha*a + (1-alpha)*b
    static GradBuf mix(double alpha, const GradBuf& a, const GradBuf& b) {
        double beta = 1.0 - alpha;
        GradBuf out(a.dim, a.H);
        for (int i=0;i<a.dim;++i) out.center[i] = alpha*a.center[i] + beta*b.center[i];
        for (int i=0;i<a.dim;++i) out.v[i]      = alpha*a.v[i]      + beta*b.v[i];
        for (int i=0;i<a.dim;++i) out.w[i]      = alpha*a.w[i]      + beta*b.w[i];
        for (int k=0;k<=a.H;++k)  out.coefs[k]  = alpha*a.coefs[k]  + beta*b.coefs[k];
        return out;
    }

    void axpy(double lr, const GradBuf& d) {
        for (int i=0;i<dim;++i) center[i] += lr*d.center[i];
        for (int i=0;i<dim;++i) v[i]      += lr*d.v[i];
        for (int i=0;i<dim;++i) w[i]      += lr*d.w[i];
        for (int k=0;k<=H;++k)  coefs[k]  += lr*d.coefs[k];
    }
};

// ─── Funciones auxiliares ─────────────────────────────────────
inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// Polinomios de Chebyshev de 1ª especie: T_0..T_n en x
inline std::vector<double> cheby_T(double x, int n) {
    std::vector<double> T(n+1);
    T[0] = 1.0;
    if (n >= 1) T[1] = x;
    for (int k=2;k<=n;++k) T[k] = 2.*x*T[k-1] - T[k-2];
    return T;
}

// Polinomios de Chebyshev de 2ª especie: U_0..U_{n-1}
// T_k'(x) = k * U_{k-1}(x)  →  usados para dR/d(cos_θ)
inline std::vector<double> cheby_U(double x, int n) {
    if (n < 0) return {};
    std::vector<double> U(n+1);
    U[0] = 1.0;
    if (n >= 1) U[1] = 2.*x;
    for (int k=2;k<=n;++k) U[k] = 2.*x*U[k-1] - U[k-2];
    return U;
}

// ─── Gradiente de D_S respecto a todos los parámetros ─────────
// Escalado por dloss_dDs = ∂Loss/∂D_S para esta muestra
GradBuf grad_signed_distance(
    const nsga2::AngularManifold& m,
    const std::vector<double>& x,
    double dloss_dDs)
{
    int dim = m.dim, H = m.n_harmonics;
    GradBuf g(dim, H);

    // Vectores auxiliares
    std::vector<double> u(dim);
    double r2=0., v2=0., dot_uv=0.;
    for (int i=0;i<dim;++i) {
        u[i]    = x[i] - m.center[i];
        r2     += m.w[i]*u[i]*u[i];
        v2     += m.w[i]*m.v[i]*m.v[i];
        dot_uv += m.w[i]*u[i]*m.v[i];
    }
    double r      = std::sqrt(r2);
    double v_norm = std::sqrt(v2);

    // Caso degenerado: x ≈ centro o v degenerado
    if (r < 1e-10 || v_norm < 1e-10) {
        // D_S ≈ r - coefs[0]
        if (r > 1e-10)
            for (int i=0;i<dim;++i)
                g.center[i] = dloss_dDs * (-m.w[i]*u[i]/r);
        g.coefs[0] = dloss_dDs * (-1.0);
        return g;
    }

    double cos_theta = std::clamp(dot_uv / (r*v_norm), -1.0, 1.0);

    // T_k(cos_θ) y dR/d(cos_θ) = Σ a_k * k * U_{k-1}(cos_θ)
    auto T = cheby_T(cos_theta, H);
    double dR_dcos = 0.0;
    if (H >= 1) {
        auto U = cheby_U(cos_theta, H-1);
        for (int k=1;k<=H;++k)
            dR_dcos += m.coefs[k] * k * U[k-1];
    }

    // ∂D_S/∂coefs[k] = -T_k(cos_θ)
    for (int k=0;k<=H;++k)
        g.coefs[k] = dloss_dDs * (-T[k]);

    // ∂D_S/∂center[i]:
    //   ∂r/∂c_i     = -w_i*u_i / r
    //   ∂cos/∂c_i   = -w_i*v_i/(r*v_norm) + cos*w_i*u_i/r²
    //   ∂D_S/∂c_i   = ∂r/∂c_i - dR_dcos * ∂cos/∂c_i
    for (int i=0;i<dim;++i) {
        double dr_dc   = -m.w[i]*u[i] / r;
        double dcos_dc = (-m.w[i]*m.v[i]) / (r*v_norm)
                        + cos_theta * m.w[i]*u[i] / r2;
        g.center[i] = dloss_dDs * (dr_dc - dR_dcos*dcos_dc);
    }

    // ∂D_S/∂v[i]:
    //   ∂r/∂v_i   = 0
    //   ∂cos/∂v_i = w_i*u_i/(r*v_norm) - cos*w_i*v_i/v²
    //   ∂D_S/∂v_i = -dR_dcos * ∂cos/∂v_i
    for (int i=0;i<dim;++i) {
        double dcos_dv = m.w[i]*u[i] / (r*v_norm)
                       - cos_theta*m.w[i]*m.v[i] / v2;
        g.v[i] = dloss_dDs * (-dR_dcos*dcos_dv);
    }

    // ∂D_S/∂w[i]:
    //   ∂r/∂w_i   = u_i² / (2r)
    //   ∂cos/∂w_i = u_i*v_i/(r*v_norm) - cos*(u_i²/(2r²) + v_i²/(2v²))
    //   ∂D_S/∂w_i = ∂r/∂w_i - dR_dcos * ∂cos/∂w_i
    for (int i=0;i<dim;++i) {
        double dr_dw   = u[i]*u[i] / (2.0*r);
        double dcos_dw = u[i]*m.v[i] / (r*v_norm)
                       - cos_theta*(u[i]*u[i]/(2.0*r2) + m.v[i]*m.v[i]/(2.0*v2));
        g.w[i] = dloss_dDs * (dr_dw - dR_dcos*dcos_dw);
    }

    return g;
}

// ─── Calcula g1 = ∇O1 y g2 = ∇O2 sobre un mini-batch ─────────
std::pair<GradBuf, GradBuf> compute_gradients(
    const nsga2::AngularManifold& m,
    const nsga2::Dataset& data,
    const std::vector<int>& batch,   // vacío = full batch
    double lambda_l1)
{
    int dim = m.dim, H = m.n_harmonics;
    GradBuf g1(dim,H), g2(dim,H);
    g1.zero(); g2.zero();

    int n = batch.empty() ? data.size() : (int)batch.size();
    double inv_n = 1.0 / n;

    // ── g1: Logistic Loss sobre el batch ─────────────────────
    auto process = [&](const nsga2::Sample& s) {
        double ds  = m.signed_distance(s.x);
        // Convención: D_S < 0 = interior (clase positiva, label=+1)
        //             D_S >= 0 = exterior (clase negativa, label=-1)
        // Clasificación correcta cuando y * D_S < 0
        // Loss = log(1 + exp(y * ds))  (penaliza cuando y*ds > 0)
        // ∂Loss/∂ds = y * σ(y * ds)
        double y   = (s.label == 1) ? 1.0 : -1.0;
        double dL  = y * sigmoid(y * ds) * inv_n;
        auto   gp  = grad_signed_distance(m, s.x, dL);
        for (int i=0;i<dim;++i) g1.center[i] += gp.center[i];
        for (int i=0;i<dim;++i) g1.v[i]      += gp.v[i];
        for (int i=0;i<dim;++i) g1.w[i]      += gp.w[i];
        for (int k=0;k<=H;++k)  g1.coefs[k]  += gp.coefs[k];
    };

    if (batch.empty()) {
        for (auto& s : data.samples) process(s);
    } else {
        for (int idx : batch)       process(data.samples[idx]);
    }

    // ── g2: Complejidad = |coefs| (energía) + L1 en w ────────
    double inv_H1  = 1.0 / (H + 1.0);
    double inv_dim = lambda_l1 / m.dim;
    for (int k=0;k<=H;++k)
        g2.coefs[k] = (m.coefs[k] > 0 ? 1.0 : (m.coefs[k] < 0 ? -1.0 : 0.0)) * inv_H1;
    for (int i=0;i<dim;++i)
        g2.w[i] = inv_dim;
    // center y v no aparecen en O2

    return {g1, g2};
}

// ─── Solver MGDA cerrado (2 objetivos) ────────────────────────
// Minimiza ||α·g1 + (1-α)·g2||² sobre α ∈ [0,1]
// Solución KKT: α* = (g2·g2 - g1·g2) / (g1·g1 - 2·g1·g2 + g2·g2)
double solve_mgda(const GradBuf& g1, const GradBuf& g2) {
    double a   = g1.dot(g1);
    double b   = g1.dot(g2);
    double c   = g2.dot(g2);
    double den = a - 2.0*b + c;
    if (std::abs(den) < 1e-15) return 0.5; // gradientes casi idénticos
    return std::clamp((c - b) / den, 0.0, 1.0);
}

// ─── Actualización con proyecciones ──────────────────────────
void apply_update(nsga2::AngularManifold& m, const GradBuf& d, double lr) {
    for (int i=0;i<m.dim;++i) m.center[i] -= lr*d.center[i];
    for (int i=0;i<m.dim;++i) m.v[i]      -= lr*d.v[i];
    for (int i=0;i<m.dim;++i) {
        m.w[i] -= lr*d.w[i];
        m.w[i]  = std::clamp(m.w[i], 1e-6, 2.0); // w_i ∈ (0, 2]
    }
    for (int k=0;k<=m.n_harmonics;++k)
        m.coefs[k] -= lr*d.coefs[k];
    m.coefs[0] = std::max(m.coefs[0], 0.05); // radio base siempre positivo
    m.normalize_v();                           // proyección sobre esfera
}

// ─── Evaluación completa de objetivos (sin gradiente) ─────────
std::pair<double,double> eval_objectives(
    const nsga2::AngularManifold& m,
    const nsga2::Dataset& data,
    double lambda_l1)
{
    // O1: tasa de error 0-1
    int n_wrong = 0;
    for (auto& s : data.samples) {
        bool pred  = m.is_inside(s.x);
        bool truth = (s.label == 1);
        if (pred != truth) ++n_wrong;
    }
    double error = (double)n_wrong / data.size();

    // O2: energía de coeficientes + L1 de pesos
    double ce = 0.0;
    for (auto a : m.coefs) ce += std::abs(a);
    ce /= m.coefs.size();
    double wl1 = 0.0;
    for (auto wi : m.w) wl1 += wi;
    wl1 /= m.dim;
    double complexity = (1.0-lambda_l1)*ce + lambda_l1*wl1;

    return {error, complexity};
}

// ─── Resultado del entrenamiento ──────────────────────────────
struct MGDAResult {
    nsga2::AngularManifold         manifold;
    std::vector<TrajectoryPoint>   trajectory;
};

// ─── Loop principal MGDA ──────────────────────────────────────
MGDAResult mgda_train(
    const nsga2::Dataset& binary_data,
    const std::vector<double>& class_center,
    const MGDAConfig& cfg = {})
{
    int dim = (int)binary_data.samples[0].x.size();
    int n   = binary_data.size();

    // Inicializar manifold cerca del centroide de la clase
    nsga2::AngularManifold m(dim, cfg.n_harmonics);
    m.randomize(0.3);
    m.set_center_near(class_center, 0.05);

    MGDAResult result;
    result.trajectory.reserve(cfg.n_iter / cfg.log_every + 2);

    double lr = cfg.lr;

    // Mini-batch: índices aleatorios con LCG (sin dependencias externas)
    uint32_t rng_state = 12345u;
    auto rand_batch = [&]() -> std::vector<int> {
        if (cfg.batch_size <= 0 || cfg.batch_size >= n) return {};
        std::vector<int> idx(cfg.batch_size);
        for (auto& i : idx) {
            rng_state = rng_state * 1664525u + 1013904223u;
            i = (int)(rng_state % (uint32_t)n);
        }
        return idx;
    };

    for (int iter = 0; iter < cfg.n_iter; ++iter) {
        auto batch = rand_batch();

        // 1. Gradientes analíticos
        auto [g1, g2] = compute_gradients(m, binary_data, batch, cfg.lambda_l1);

        // 2. Resolver QP → coeficiente de mezcla α
        double alpha = solve_mgda(g1, g2);

        // 3. Dirección combinada d = α·g1 + (1-α)·g2
        auto d = GradBuf::mix(alpha, g1, g2);
        double gn = std::sqrt(d.norm_sq());

        // 4. Log + eval completa periódicamente
        if (iter % cfg.log_every == 0 || iter == cfg.n_iter-1) {
            auto [err, cpx] = eval_objectives(m, binary_data, cfg.lambda_l1);
            result.trajectory.push_back({iter, err, cpx, alpha, gn});
            if (cfg.verbose)
                std::cout << "  [MGDA] iter=" << std::setw(5) << iter
                          << "  err=" << std::fixed << std::setprecision(4) << err
                          << "  cpx=" << cpx
                          << "  α="   << std::setprecision(3) << alpha
                          << "  ||d||=" << std::scientific << std::setprecision(2) << gn
                          << std::defaultfloat << "\n";
        }

        // 5. Convergencia: ||d|| < tol → Pareto point detectado
        if (gn < cfg.tol && iter > 50) {
            if (cfg.verbose)
                std::cout << "  [MGDA] ✓ Pareto point at iter=" << iter
                          << " (||d||=" << gn << ")\n";
            auto [err, cpx] = eval_objectives(m, binary_data, cfg.lambda_l1);
            result.trajectory.push_back({iter, err, cpx, alpha, gn});
            break;
        }

        // 6. Actualizar θ ← θ - lr · d
        apply_update(m, d, lr);
        lr *= cfg.lr_decay;
    }

    result.manifold = m;
    return result;
}

} // namespace mgda
