#pragma once
// ============================================================
// include/sh_manifold.hpp — Spherical Harmonic Manifold (SHM)
// ============================================================
//
// CONCEPTO:
//   Frontera de decisión = superficie star-shaped en R^dim
//   definida por un centroide c y un radio angular:
//
//     r*(θ,φ) = Σ_{l=0}^{L} Σ_{m=-l}^{l}  c_{l,m} · Y_l^m(θ,φ)
//
//   donde Y_l^m son los Armónicos Esféricos REALES en S² ⊂ R³,
//   y (θ,φ) se obtienen proyectando (x - c) sobre tres vectores
//   ortonormales {v1, v2, v3} aprendidos en R^dim.
//
// INFERENCIA (O(dim + L²)):
//   1. p = [<(x-c)·w, v1>, <(x-c)·w, v2>, <(x-c)·w, v3>]
//   2. r = ||p||
//   3. (θ, φ) = ángulos esféricos de p
//   4. r* = Σ_{l,m} c_{l,m} · Y_l^m(θ, φ)
//   5. D_S = r - r*  →  <0 = dentro, ≥0 = fuera
//
// CROMOSOMA:
//   center (dim) | v1 (dim) | v2 (dim) | v3 (dim) | w (dim) | sh_coefs ((L+1)²)
// ============================================================
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "utils.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nsga2 {

// ─── Polinomio de Legendre asociado P_l^m(x) — sin fase C-S ─
inline double assoc_legendre(int l, int m, double x) {
    // Calcular P_m^m(x) = (2m-1)!! * (1-x²)^(m/2)
    double pmm = 1.0;
    double sx  = std::sqrt(std::max(0.0, 1.0 - x * x));
    for (int i = 1; i <= m; ++i) pmm *= (2.0*i - 1.0) * sx;

    if (l == m) return pmm;
    double pmm1 = x * (2*m + 1) * pmm;
    if (l == m + 1) return pmm1;

    double plm = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        plm  = (x * (2*ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m);
        pmm  = pmm1;
        pmm1 = plm;
    }
    return plm;
}

// Factor de normalización N_l^|m|
inline double sh_norm(int l, int m) {
    double val = (2.0*l + 1.0) / (4.0 * M_PI);
    for (int i = l - m + 1; i <= l + m; ++i) val /= i;
    return std::sqrt(val);
}

// Armónico esférico real Y_l^m(θ,φ)
inline double real_sh(int l, int m, double theta, double phi) {
    int am = std::abs(m);
    double N   = sh_norm(l, am);
    double Plm = assoc_legendre(l, am, std::cos(theta));
    if (m > 0) return std::sqrt(2.0) * N * Plm * std::cos( m * phi);
    if (m < 0) return std::sqrt(2.0) * N * Plm * std::sin(am * phi);
    return N * Plm;
}

inline int sh_index(int l, int m) { return l*l + l + m; }
inline int sh_total(int L) { return (L + 1) * (L + 1); }

// ─── SphericalHarmonicManifold ────────────────────────────────
struct SphericalHarmonicManifold {

    int dim;
    int n_harmonics;  // grado máximo L

    std::vector<double> center;
    std::vector<double> v1, v2, v3;  // base de proyección aprendida
    std::vector<double> w;           // feature weights
    std::vector<double> coefs;       // c_{l,m}, tamaño (L+1)²

    static constexpr double AMP_MAX   = 1.0;
    static constexpr double CENTER_LO = 0.0;
    static constexpr double CENTER_HI = 1.0;

    SphericalHarmonicManifold() : dim(0), n_harmonics(0) {}

    SphericalHarmonicManifold(int dim_, int L)
        : dim(dim_), n_harmonics(L),
          center(dim_, 0.5),
          v1(dim_, 0.0), v2(dim_, 0.0), v3(dim_, 0.0),
          w(dim_, 1.0),
          coefs(sh_total(L), 0.0)
    {
        assert(dim_ >= 2 && L >= 0);
        // Ejes canónicos iniciales
        v1[0] = 1.0;
        if (dim_ > 1) v2[1] = 1.0;
        if (dim_ > 2) v3[2] = 1.0; else v3[0] = 1.0;
        // Radio inicial ~ 0.3 (esfera pequeña)
        double y00 = real_sh(0, 0, 0.0, 0.0);
        coefs[0] = (std::abs(y00) > 1e-12) ? 0.3 / y00 : 0.3;
    }

    bool valid() const {
        return dim > 0 && n_harmonics >= 0
            && (int)center.size() == dim
            && (int)v1.size() == dim && (int)v2.size() == dim && (int)v3.size() == dim
            && (int)w.size() == dim
            && (int)coefs.size() == sh_total(n_harmonics);
    }

    // Gram-Schmidt ponderado por w
    void orthonormalize() {
        auto wdot = [&](const std::vector<double>& a, const std::vector<double>& b) {
            double s = 0.0;
            for (int i = 0; i < dim; ++i) s += std::max(w[i], 0.0) * a[i] * b[i];
            return s;
        };
        auto wnorm = [&](std::vector<double>& v) {
            double n = std::sqrt(std::max(1e-24, wdot(v, v)));
            for (auto& x : v) x /= n;
        };
        auto wproj = [&](std::vector<double>& u, const std::vector<double>& e) {
            double p = wdot(u, e);
            for (int i = 0; i < dim; ++i) u[i] -= p * e[i];
        };
        wnorm(v1);
        wproj(v2, v1); wnorm(v2);
        wproj(v3, v1); wproj(v3, v2); wnorm(v3);
    }

    // Proyecta (x-c) sobre la base {v1, v2, v3}
    void project(const std::vector<double>& x,
                 double& p1, double& p2, double& p3) const {
        p1 = p2 = p3 = 0.0;
        for (int i = 0; i < dim; ++i) {
            double ui = (x[i] - center[i]) * std::max(w[i], 0.0);
            p1 += ui * v1[i];
            p2 += ui * v2[i];
            p3 += ui * v3[i];
        }
    }

    double evaluate_radius(double theta, double phi) const {
        double r = 0.0;
        for (int l = 0; l <= n_harmonics; ++l)
            for (int m = -l; m <= l; ++m)
                r += coefs[sh_index(l, m)] * real_sh(l, m, theta, phi);
        return r;
    }

    double signed_distance(const std::vector<double>& x) const {
        double p1, p2, p3;
        project(x, p1, p2, p3);

        double r = std::sqrt(p1*p1 + p2*p2 + p3*p3);
        if (r < 1e-12) return -std::abs(coefs[0]);

        double cos_theta = std::clamp(p3 / r, -1.0, 1.0);
        double theta = std::acos(cos_theta);
        double phi   = std::atan2(p2, p1);

        double r_star = evaluate_radius(theta, phi);
        return r - r_star;
    }

    bool is_inside(const std::vector<double>& x) const {
        return signed_distance(x) < 0.0;
    }

    // ─── Complejidad ──────────────────────────────────────────
    double arc_length(int = 0) const {
        double e = 0.0;
        for (int l = 0; l <= n_harmonics; ++l)
            for (int m = -l; m <= l; ++m)
                e += (l + 1.0) * std::abs(coefs[sh_index(l, m)]);
        return e;
    }
    double coef_energy() const {
        double e = 0.0;
        for (auto v : coefs) e += std::abs(v);
        return e;
    }

    // ─── Operadores genéticos ─────────────────────────────────

    void add_harmonic() {
        ++n_harmonics;
        for (int m = -n_harmonics; m <= n_harmonics; ++m)
            coefs.push_back(rand_double(-0.05, 0.05));
    }

    bool remove_harmonic() {
        if (n_harmonics <= 0) return false;
        for (int m = -n_harmonics; m <= n_harmonics; ++m)
            coefs.pop_back();
        --n_harmonics;
        return true;
    }

    void randomize(double scale = 0.5) {
        for (int i = 0; i < dim; ++i) {
            center[i] = rand_double(CENTER_LO, CENTER_HI);
            v1[i] = rand_normal(0.0, 1.0);
            v2[i] = rand_normal(0.0, 1.0);
            v3[i] = rand_normal(0.0, 1.0);
            w[i]  = rand_double(0.0, 1.0);
        }
        orthonormalize();
        for (auto& c : coefs) c = rand_double(-scale * 0.05, scale * 0.05);
        coefs[0] = std::abs(rand_double(0.1, scale));
    }

    void set_center_near(const std::vector<double>& c, double noise = 0.05) {
        for (int i = 0; i < dim; ++i)
            center[i] = c[i] + rand_normal(0.0, noise);
    }

    int n_params() const { return 4*dim + (int)coefs.size(); }
};

} // namespace nsga2
