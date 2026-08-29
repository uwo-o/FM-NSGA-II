#pragma once
// ============================================================
// include/mgda_kernels.hpp — Gradientes MGDA para los 3 kernels
// ============================================================
//
// Implementa compute_gradients() para cada tipo de manifold:
//
//  [1] AngularManifold  → gradiente ANALÍTICO (chain rule completo)
//  [2] GAMManifold      → gradiente ANALÍTICO (d(x) es lineal en params)
//  [3] FourierManifold  → gradiente NUMÉRICO  (diferencias finitas + Envelope Theorem)
//                         La proyección t* hace el gradiente analítico
//                         de centro difícil; finitas son ~O(4·N·dim) evaluaciones.
//
// Interfaz unificada para mgda_generic.hpp:
//   GradBuf<M> compute_g1(m, data, batch, eps)   → gradiente de O1 (pérdida)
//   GradBuf<M> compute_g2(m, lambda_l1)           → gradiente de O2 (complejidad)
// ============================================================
#include <vector>
#include <cmath>
#include <cassert>
#include "angular_manifold.hpp"
#include "gam_manifold.hpp"
#include "manifold.hpp"
#include "dataset.hpp"

namespace mgda {

// ─────────────────────────────────────────────────────────────
//  Utilidades de pérdida mejoradas
// ─────────────────────────────────────────────────────────────

inline double sigmoid(double z) {
    // Numéricamente estable (evita overflow)
    return z >= 0 ? 1.0/(1.+std::exp(-z)) : std::exp(z)/(1.+std::exp(z));
}

// softplus estable: log(1 + exp(z))
inline double softplus(double z) {
    if (z > 30.)  return z;                           // exp overflow region
    if (z < -30.) return std::exp(z);                 // numerically 0
    return std::log(1. + std::exp(z));
}

// ─────────────────────────────────────────────────────────────
//  Focal Loss (Lin et al., 2017)
//
//  Loss_i = sigma(y*ds)^gamma  *  log(1 + exp(y*ds))
//
//  dLoss/dds = y * sigma(y*ds)^gamma * (gamma*(1-sigma)*softplus(y*ds) + sigma)
//
//  gamma=0  -->  logistic clásica
//  gamma=2  -->  enfoque en ejemplos difíciles (near-boundary)
//
//  Efecto anti-mínimos-locales: cuando un punto está muy mal
//  clasificado (sigma ≈ 1), el factor sigma^gamma AMPLIFICA su
//  gradiente en lugar de aplastarlo como hace la logistica saturada.
// ─────────────────────────────────────────────────────────────
inline double focal_grad_scale(double ds, double y, double gamma) {
    double yd  = y * ds;
    double s   = sigmoid(yd);
    double sp  = softplus(yd);
    if (gamma == 0.) return y * s;
    double sg  = std::pow(s, gamma);
    return y * sg * (gamma * (1. - s) * sp + s);
}

// ─── Chebyshev helpers (usados por Angular) ──────────────────
inline std::vector<double> cheby_T(double x, int n) {
    std::vector<double> T(n+1); T[0]=1.;
    if (n>=1) T[1]=x;
    for (int k=2;k<=n;++k) T[k]=2.*x*T[k-1]-T[k-2];
    return T;
}
inline std::vector<double> cheby_U(double x, int n) {
    if (n<0) return {};
    std::vector<double> U(n+1); U[0]=1.;
    if (n>=1) U[1]=2.*x;
    for (int k=2;k<=n;++k) U[k]=2.*x*U[k-1]-U[k-2];
    return U;
}

// ============================================================
// Parámetros empaquetados en un vector flat con slices
// por tipo de manifold (para GradBuf genérico).
// ============================================================

// ─────────────────────────────────────────────────────────────
//  ANGULAR MANIFOLD: gradiente analítico completo
// ─────────────────────────────────────────────────────────────

// Gradiente de D_S respecto a params del AngularManifold,
// escalado por dloss_dDs = ∂Loss/∂D_S para esta muestra.
// Retorna vector flat [center(dim), v(dim), w(dim), coefs(H+1)]
inline std::vector<double> angular_grad_ds(
    const nsga2::AngularManifold& m,
    const std::vector<double>& x,
    double dloss_dDs)
{
    int dim = m.dim, H = m.n_harmonics;
    std::vector<double> g(3*dim + H+1, 0.0);
    // slices: center[0..dim), v[dim..2dim), w[2dim..3dim), coefs[3dim..3dim+H+1)
    auto gc = g.data();
    auto gv = g.data() + dim;
    auto gw = g.data() + 2*dim;
    auto gk = g.data() + 3*dim;

    std::vector<double> u(dim);
    double r2=0., v2=0., dot_uv=0.;
    for (int i=0;i<dim;++i) {
        u[i]    = x[i]-m.center[i];
        r2     += m.w[i]*u[i]*u[i];
        v2     += m.w[i]*m.v[i]*m.v[i];
        dot_uv += m.w[i]*u[i]*m.v[i];
    }
    double r = std::sqrt(r2), vn = std::sqrt(v2);
    if (r < 1e-10 || vn < 1e-10) {
        if (r > 1e-10)
            for (int i=0;i<dim;++i) gc[i] = dloss_dDs*(-m.w[i]*u[i]/r);
        gk[0] = dloss_dDs*(-1.0);
        return g;
    }
    double cos_t = std::clamp(dot_uv/(r*vn), -1.0, 1.0);
    auto T = cheby_T(cos_t, H);
    double dR = 0.0;
    if (H>=1) {
        auto U = cheby_U(cos_t, H-1);
        for (int k=1;k<=H;++k) dR += m.coefs[k]*k*U[k-1];
    }
    // ∂D_S/∂coefs[k] = -T_k
    for (int k=0;k<=H;++k) gk[k] = dloss_dDs*(-T[k]);
    // ∂D_S/∂center[i]
    for (int i=0;i<dim;++i) {
        double dr_dc   = -m.w[i]*u[i]/r;
        double dcos_dc = (-m.w[i]*m.v[i])/(r*vn) + cos_t*m.w[i]*u[i]/r2;
        gc[i] = dloss_dDs*(dr_dc - dR*dcos_dc);
    }
    // ∂D_S/∂v[i]
    for (int i=0;i<dim;++i) {
        double dcos_dv = m.w[i]*u[i]/(r*vn) - cos_t*m.w[i]*m.v[i]/v2;
        gv[i] = dloss_dDs*(-dR*dcos_dv);
    }
    // ∂D_S/∂w[i]
    for (int i=0;i<dim;++i) {
        double dr_dw   = u[i]*u[i]/(2.0*r);
        double dcos_dw = u[i]*m.v[i]/(r*vn)
                       - cos_t*(u[i]*u[i]/(2.0*r2)+m.v[i]*m.v[i]/(2.0*v2));
        gw[i] = dloss_dDs*(dr_dw - dR*dcos_dw);
    }
    return g;
}

// g1 (Error) y g2 (Complejidad) para AngularManifold
inline std::pair<std::vector<double>, std::vector<double>>
angular_gradients(const nsga2::AngularManifold& m,
                  const nsga2::Dataset& data,
                  const std::vector<int>& batch,
                  double lambda_l1,
                  double focal_gamma = 0.)
{
    int dim=m.dim, H=m.n_harmonics;
    int P = 3*dim + H+1;
    std::vector<double> g1(P,0.), g2(P,0.);
    int n = batch.empty() ? data.size() : (int)batch.size();
    double inv_n = 1.0/n;
    auto process = [&](const nsga2::Sample& s) {
        double ds = m.signed_distance(s.x);
        double y  = (s.label==1)?1.:-1.;
        double dL = focal_grad_scale(ds, y, focal_gamma) * inv_n;
        auto gp = angular_grad_ds(m, s.x, dL);
        for (int j=0;j<P;++j) g1[j] += gp[j];
    };
    if (batch.empty()) for (auto& s:data.samples) process(s);
    else               for (int idx:batch) process(data.samples[idx]);

    double inv_H1  = 1.0/(H+1.0);
    double inv_dim = lambda_l1/m.dim;
    auto gk = g2.data()+3*dim;
    auto gw = g2.data()+2*dim;
    for (int k=0;k<=H;++k)
        gk[k] = (m.coefs[k]>0?1.:(m.coefs[k]<0?-1.:0.))*inv_H1;
    for (int i=0;i<dim;++i) gw[i] = inv_dim;
    return {g1, g2};
}

// Aplicar actualización a AngularManifold desde vector flat
inline void angular_apply(nsga2::AngularManifold& m,
                          const std::vector<double>& d, double lr)
{
    int dim=m.dim, H=m.n_harmonics;
    for (int i=0;i<dim;++i) m.center[i] -= lr*d[i];
    for (int i=0;i<dim;++i) m.v[i]      -= lr*d[dim+i];
    for (int i=0;i<dim;++i) {
        m.w[i] -= lr*d[2*dim+i];
        m.w[i]  = std::clamp(m.w[i], 1e-6, 2.0);
    }
    for (int k=0;k<=H;++k) m.coefs[k] -= lr*d[3*dim+k];
    m.coefs[0] = std::max(m.coefs[0], 0.05);
    m.normalize_v();
}

// ─────────────────────────────────────────────────────────────
//  GAM MANIFOLD: gradiente analítico (d(x) lineal en params)
// ─────────────────────────────────────────────────────────────
// Layout flat: [beta0(1), w(dim), a_{0,1},b_{0,1},...,a_{D-1,H},b_{D-1,H}]

inline std::pair<std::vector<double>, std::vector<double>>
gam_gradients(const nsga2::GAMManifold& m,
              const nsga2::Dataset& data,
              const std::vector<int>& batch,
              double lambda_l1,
              double focal_gamma = 0.)
{
    int dim=m.dim, H=m.n_harmonics;
    int P = 1 + dim + dim*2*H;  // beta0 + w + coefs
    std::vector<double> g1(P,0.), g2(P,0.);
    int n = batch.empty() ? data.size() : (int)batch.size();
    double inv_n = 1.0/n;

    auto process = [&](const nsga2::Sample& s) {
        double ds = m.signed_distance(s.x);
        double y  = (s.label==1)?1.:-1.;
        double dL = focal_grad_scale(ds, y, focal_gamma) * inv_n;  // focal / logistic

        // ∂ds/∂beta0 = 1
        g1[0] += dL;
        for (int i=0;i<dim;++i) {
            double fi = m.eval_fi(i, s.x[i]);
            // ∂ds/∂w_i = f_i(x_i)
            g1[1+i] += dL * fi;
            // ∂ds/∂a_{i,k} = w_i * cos(k*pi*x_i)
            // ∂ds/∂b_{i,k} = w_i * sin(k*pi*x_i)
            int base = 1 + dim + i*2*H;
            for (int k=1;k<=H;++k) {
                double kpx = k * M_PI * s.x[i];
                g1[base + (k-1)*2    ] += dL * m.w[i] * std::cos(kpx);
                g1[base + (k-1)*2 + 1] += dL * m.w[i] * std::sin(kpx);
            }
        }
    };
    if (batch.empty()) for (auto& s:data.samples) process(s);
    else               for (int idx:batch) process(data.samples[idx]);

    // O2: roughness = Σ k²(a²+b²) + L1 de w + |beta0|
    double inv_total = 1.0/(dim*2*H+1.0);
    g2[0] = (m.beta0>0?1.:(m.beta0<0?-1.:0.))*inv_total;  // |beta0| penalty
    for (int i=0;i<dim;++i) g2[1+i] = lambda_l1/dim;       // L1 en w
    // Subgradiente de k²(a²+b²) → 2k²a y 2k²b
    int base0 = 1+dim;
    for (int i=0;i<dim;++i) {
        int base = base0 + i*2*H;
        for (int k=1;k<=H;++k) {
            double scale = 2.0*k*k*inv_total;
            g2[base+(k-1)*2    ] = scale * m.coefs[i*2*H+(k-1)*2    ];
            g2[base+(k-1)*2 + 1] = scale * m.coefs[i*2*H+(k-1)*2 + 1];
        }
    }
    return {g1, g2};
}

inline void gam_apply(nsga2::GAMManifold& m,
                      const std::vector<double>& d, double lr)
{
    int dim=m.dim, H=m.n_harmonics;
    m.beta0 -= lr*d[0];
    for (int i=0;i<dim;++i) {
        m.w[i] -= lr*d[1+i];
        m.w[i]  = std::clamp(m.w[i], 0.0, 2.0);
    }
    for (int j=0;j<dim*2*H;++j)
        m.coefs[j] -= lr*d[1+dim+j];
}

// ─────────────────────────────────────────────────────────────
//  FOURIER MANIFOLD: gradiente numérico (diferencias finitas)
//  Layout flat: [center(dim), w(dim), coefs(2*N*dim)]
// ─────────────────────────────────────────────────────────────

// Evalúa la pérdida Focal promedio sobre un batch/full dataset
inline double fourier_loss(const nsga2::FourierManifold& m,
                           const nsga2::Dataset& data,
                           const std::vector<int>& batch,
                           double focal_gamma = 0.)
{
    double loss = 0.;
    int n = batch.empty() ? data.size() : (int)batch.size();
    auto eval = [&](const nsga2::Sample& s) {
        double ds = m.signed_distance(s.x);
        double y  = (s.label==1)?1.:-1.;
        double yd = y * ds;
        double sg = std::pow(sigmoid(yd), focal_gamma);
        loss += sg * softplus(yd);   // Focal: s^gamma * log(1+exp(y*ds))
    };
    if (batch.empty()) for (auto& s:data.samples) eval(s);
    else               for (int idx:batch) eval(data.samples[idx]);
    return loss/n;
}

inline std::pair<std::vector<double>, std::vector<double>>
fourier_gradients(nsga2::FourierManifold& m,
                  const nsga2::Dataset& data,
                  const std::vector<int>& batch,
                  double lambda_l1,
                  double eps = 1e-5,
                  double focal_gamma = 0.)
{
    int dim=m.dim, N=m.n_harmonics;
    int Pc = 2*N*dim;
    int P  = dim + dim + Pc;
    std::vector<double> g1(P,0.), g2(P,0.);

    auto loss_fn = [&]{ return fourier_loss(m, data, batch, focal_gamma); };

    for (int i=0;i<dim;++i) {
        m.center[i] += eps; double fp=loss_fn();
        m.center[i] -= 2*eps; double fm=loss_fn();
        m.center[i] += eps;
        g1[i] = (fp-fm)/(2*eps);
    }
    for (int i=0;i<dim;++i) {
        m.w[i] += eps; double fp=loss_fn();
        m.w[i] -= 2*eps; double fm=loss_fn();
        m.w[i] += eps;
        g1[dim+i] = (fp-fm)/(2*eps);
    }
    for (int j=0;j<Pc;++j) {
        m.coefs[j] += eps; double fp=loss_fn();
        m.coefs[j] -= 2*eps; double fm=loss_fn();
        m.coefs[j] += eps;
        g1[2*dim+j] = (fp-fm)/(2*eps);
    }

    double inv = lambda_l1/dim;
    for (int i=0;i<dim;++i) g2[dim+i] = inv;
    for (int j=0;j<Pc;++j) {
        m.coefs[j] += eps; double ap=m.arc_length(30);
        m.coefs[j] -= 2*eps; double am=m.arc_length(30);
        m.coefs[j] += eps;
        g2[2*dim+j] = (ap-am)/(2*eps) / 10.0;
    }
    return {g1, g2};
}

inline void fourier_apply(nsga2::FourierManifold& m,
                          const std::vector<double>& d, double lr)
{
    int dim=m.dim, N=m.n_harmonics, Pc=2*N*dim;
    for (int i=0;i<dim;++i) m.center[i] -= lr*d[i];
    for (int i=0;i<dim;++i) {
        m.w[i] -= lr*d[dim+i];
        m.w[i]  = std::clamp(m.w[i], 1e-6, 2.0);
    }
    for (int j=0;j<Pc;++j) m.coefs[j] -= lr*d[2*dim+j];
}

// ─────────────────────────────────────────────────────────────
//  SOLVER MGDA GENÉRICO (vector flat)
// ─────────────────────────────────────────────────────────────
inline double mgda_alpha(const std::vector<double>& g1,
                         const std::vector<double>& g2) {
    double a=0.,b=0.,c=0.;
    for (int j=0;j<(int)g1.size();++j) {
        a += g1[j]*g1[j]; b += g1[j]*g2[j]; c += g2[j]*g2[j];
    }
    double den = a-2.*b+c;
    if (std::abs(den)<1e-15) return 0.5;
    return std::clamp((c-b)/den, 0.0, 1.0);
}

inline std::vector<double> mgda_combine(double alpha,
                                        const std::vector<double>& g1,
                                        const std::vector<double>& g2) {
    std::vector<double> d(g1.size());
    for (int j=0;j<(int)g1.size();++j)
        d[j] = alpha*g1[j] + (1.-alpha)*g2[j];
    return d;
}

inline double vec_norm(const std::vector<double>& v) {
    double s=0.; for (auto x:v) s+=x*x; return std::sqrt(s);
}

// ─────────────────────────────────────────────────────────────
//  Adam Optimizer State
//
//  En lugar de θ ← θ - lr·d (SGD puro), usamos:
//    m_t = β1·m_{t-1} + (1-β1)·d       (media exponencial de d)
//    v_t = β2·v_{t-1} + (1-β2)·d²      (varianza exponencial de d²)
//    θ ← θ - lr · m̂_t / (√v̂_t + ε)    (actualización adaptativa)
//
//  Ventaja clave anti-mínimos-locales:
//  - Cada parámetro tiene su propio lr efectivo (normalizado por su historia)
//  - El momentum acumula dirección incluso cuando el gradiente oscila
//  - Zonas planas (gradiente ≈ 0) se superan con el momento acumulado
// ─────────────────────────────────────────────────────────────
struct AdamState {
    std::vector<double> m, v;   // primer y segundo momento
    int t = 0;                   // paso actual
    double beta1, beta2, eps_;

    AdamState() = default;
    explicit AdamState(int size, double b1=0.9, double b2=0.999, double ep=1e-8)
        : m(size,0.), v(size,0.), t(0), beta1(b1), beta2(b2), eps_(ep) {}

    // Devuelve el paso Adam (ya escalado por lr)
    std::vector<double> step(const std::vector<double>& g, double lr) {
        ++t;
        double bc1 = 1. - std::pow(beta1, t);
        double bc2 = 1. - std::pow(beta2, t);
        std::vector<double> update(g.size());
        for (int i=0;i<(int)g.size();++i) {
            m[i] = beta1*m[i] + (1.-beta1)*g[i];
            v[i] = beta2*v[i] + (1.-beta2)*g[i]*g[i];
            double m_hat = m[i]/bc1;
            double v_hat = v[i]/bc2;
            update[i] = lr * m_hat / (std::sqrt(v_hat) + eps_);
        }
        return update;
    }

    void reset() { std::fill(m.begin(),m.end(),0.); std::fill(v.begin(),v.end(),0.); t=0; }
};

// ─────────────────────────────────────────────────────────────
//  Cosine Annealing con Warm Restarts (SGDR — Loshchilov 2016)
//
//  lr(t) = lr_min + 0.5*(lr_max-lr_min)*(1 + cos(π*(t%T)/T))
//
//  En cada ciclo T el lr sube bruscamente (warm restart) y luego
//  desciende suavemente. Esto permite que Adam escape de mínimos
//  locales periódicamente sin necesidad de ruido explícito.
// ─────────────────────────────────────────────────────────────
inline double cosine_lr(int t, double lr_max, double lr_min, int T) {
    int t_mod = t % T;
    return lr_min + 0.5*(lr_max - lr_min)*(1. + std::cos(M_PI*t_mod/(double)T));
}

} // namespace mgda
