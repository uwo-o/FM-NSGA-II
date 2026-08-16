#pragma once
// ============================================================
// include/manifold.hpp — Manifold de Fourier en R^n (variable-length)
// ============================================================
//
// REPRESENTACIÓN:
//   Curva cerrada paramétrica en R^dim:
//     γ_i(t) = c_i + Σ_{k=1}^{N} [ a_{ik}·cos(kt) + b_{ik}·sin(kt) ]
//   donde t ∈ [0, 2π], i = 0..dim-1
//
// CLASIFICACIÓN (signed distance):
//   Para un punto x, encontrar t* = argmin_t ||x - γ(t)||
//   d(x) = ||x - c|| - ||γ(t*) - c||
//     d < 0  →  dentro del manifold  → clase positiva
//     d ≥ 0  →  fuera del manifold   → clase negativa
//
// ESTRUCTURA DEL CROMOSOMA (flat):
//   [c_0, ..., c_{dim-1},
//    a_{0,1}, b_{0,1}, ..., a_{dim-1,1}, b_{dim-1,1},   ← harmónico k=1
//    a_{0,2}, b_{0,2}, ..., a_{dim-1,2}, b_{dim-1,2},   ← harmónico k=2
//    ...]
//   Tamaño: dim + 2*N*dim   (longitud variable según N)
//
// MUTACIÓN ERC:
//   Los coeficientes Fourier son ERC: cada uno se reemplaza por
//   un valor aleatorio de U(-AMP_MAX, AMP_MAX). El centro muta
//   con ruido Gaussiano.
// ============================================================
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>
#include "utils.hpp"

namespace nsga2 {

struct FourierManifold {
    // ─── Parámetros ──────────────────────────────────────────
    int dim;          // dimensión del espacio ambiente
    int n_harmonics;  // N (longitud variable del cromosoma)

    std::vector<double> center;  // c ∈ R^dim  (tamaño: dim)
    std::vector<double> coefs;   // Fourier coefficients (tamaño: 2*N*dim)
                                 // layout: [a_{0,k}, b_{0,k}, ..., a_{dim-1,k}, b_{dim-1,k}]
                                 //         repeated for k=1..N

    // ─── Parámetros de configuración ─────────────────────────
    static constexpr int    GRID_PTS  = 32;   // puntos de grilla para proyección
    static constexpr double AMP_MAX   = 0.5;  // rango ERC de amplitudes
    static constexpr double CENTER_LO = 0.0;  // rango ERC del centro (normalizado)
    static constexpr double CENTER_HI = 1.0;

    // ─── Constructores ────────────────────────────────────────
    FourierManifold() : dim(0), n_harmonics(0) {}

    FourierManifold(int dim_, int n_harmonics_)
        : dim(dim_), n_harmonics(n_harmonics_),
          center(dim_, 0.5),
          coefs(2 * n_harmonics_ * dim_, 0.0) {
        assert(dim_ > 0 && n_harmonics_ > 0);
    }

    bool valid() const {
        return dim > 0 && n_harmonics > 0
            && (int)center.size() == dim
            && (int)coefs.size() == 2 * n_harmonics * dim;
    }

    // ─── Acceso a coeficientes ────────────────────────────────
    // Harmónico k (1-based), dimensión i (0-based)
    double& a(int k, int i)       { return coefs[(k-1)*2*dim + 2*i];     }
    double& b(int k, int i)       { return coefs[(k-1)*2*dim + 2*i + 1]; }
    const double& a(int k, int i) const { return coefs[(k-1)*2*dim + 2*i];     }
    const double& b(int k, int i) const { return coefs[(k-1)*2*dim + 2*i + 1]; }

    // ─── Evaluación de la curva ───────────────────────────────

    // γ(t) ∈ R^dim
    std::vector<double> eval(double t) const {
        std::vector<double> p(dim);
        for (int i = 0; i < dim; ++i) {
            p[i] = center[i];
            for (int k = 1; k <= n_harmonics; ++k)
                p[i] += a(k,i) * std::cos(k * t) + b(k,i) * std::sin(k * t);
        }
        return p;
    }

    // γ'(t) ∈ R^dim (derivada)
    std::vector<double> eval_deriv(double t) const {
        std::vector<double> dp(dim, 0.0);
        for (int i = 0; i < dim; ++i)
            for (int k = 1; k <= n_harmonics; ++k)
                dp[i] += k * (-a(k,i) * std::sin(k * t) + b(k,i) * std::cos(k * t));
        return dp;
    }

    // ─── Proyección (t* más cercano) ─────────────────────────
    // Precomputa tabla cos/sin para todos los k y j de la grilla,
    // evitando recalcular cos(k*t) para cada dimensión.
    double project(const std::vector<double>& x) const {
        // Precomputar tabla: cos_tab[k-1][j], sin_tab[k-1][j]
        // k=1..N_harm, j=0..GRID_PTS-1
        const int N = n_harmonics;
        std::vector<double> cos_tab(N * GRID_PTS);
        std::vector<double> sin_tab(N * GRID_PTS);
        const double step = 2.0 * M_PI / GRID_PTS;
        for (int j = 0; j < GRID_PTS; ++j) {
            double t = j * step;
            for (int k = 1; k <= N; ++k) {
                cos_tab[(k-1)*GRID_PTS + j] = std::cos(k * t);
                sin_tab[(k-1)*GRID_PTS + j] = std::sin(k * t);
            }
        }

        // Búsqueda en grilla usando tabla
        double best_t = 0.0, best_d = std::numeric_limits<double>::max();
        for (int j = 0; j < GRID_PTS; ++j) {
            double d = 0.0;
            for (int i = 0; i < dim; ++i) {
                double pi = center[i];
                for (int k = 1; k <= N; ++k)
                    pi += a(k,i) * cos_tab[(k-1)*GRID_PTS + j]
                        + b(k,i) * sin_tab[(k-1)*GRID_PTS + j];
                double diff = x[i] - pi;
                d += diff * diff;
            }
            if (d < best_d) { best_d = d; best_t = j * step; }
        }

        // Refinamiento Golden Section en [best_t - eps, best_t + eps]
        double eps = step;
        auto f = [&](double t){ return dist2_sq(x, eval(t)); };
        return golden_section(f, best_t - eps, best_t + eps, 20);
    }

    // ─── Distancia firmada ────────────────────────────────────
    // < 0 → dentro (positivo), ≥ 0 → fuera (negativo)
    double signed_distance(const std::vector<double>& x) const {
        double t_star  = project(x);
        auto   gamma   = eval(t_star);
        double r_x     = dist2(x,     center);
        double r_gamma = dist2(gamma, center);
        return r_x - r_gamma;
    }

    bool is_inside(const std::vector<double>& x) const {
        return signed_distance(x) < 0.0;
    }

    // ─── Medidas de complejidad ───────────────────────────────

    // Longitud de arco aproximada por cuadratura
    double arc_length(int n_samples = 50) const {
        double len = 0.0;
        double dt  = 2.0 * M_PI / n_samples;
        for (int j = 0; j < n_samples; ++j) {
            auto dp = eval_deriv(j * dt);
            len += norm2(dp) * dt;
        }
        return len;
    }

    int complexity() const { return n_harmonics; }

    double coef_energy() const {
        double e = 0;
        for (auto v : coefs) e += v * v;
        return e;
    }

    // ─── Operaciones sobre el cromosoma ──────────────────────

    // Añade un nuevo harmónico con coeficientes ERC
    void add_harmonic() {
        ++n_harmonics;
        for (int i = 0; i < dim; ++i) {
            coefs.push_back(rand_double(-AMP_MAX, AMP_MAX)); // a
            coefs.push_back(rand_double(-AMP_MAX, AMP_MAX)); // b
        }
    }

    // Elimina el último harmónico (retorna false si N=1)
    bool remove_harmonic() {
        if (n_harmonics <= 1) return false;
        coefs.resize(coefs.size() - 2 * dim);
        --n_harmonics;
        return true;
    }

    // Inicialización ERC aleatoria completa
    void randomize(double scale = AMP_MAX) {
        for (int i = 0; i < dim; ++i)
            center[i] = rand_double(CENTER_LO, CENTER_HI);
        for (auto& c : coefs)
            c = rand_double(-scale, scale);
    }

    // Fija el centro aproximado al centroide de los datos positivos + ruido
    void set_center_near(const std::vector<double>& c, double noise = 0.05) {
        assert((int)c.size() == dim);
        for (int i = 0; i < dim; ++i)
            center[i] = c[i] + rand_normal(0.0, noise);
    }

    // Número total de parámetros del cromosoma
    int n_params() const { return dim + (int)coefs.size(); }
};

} // namespace nsga2
