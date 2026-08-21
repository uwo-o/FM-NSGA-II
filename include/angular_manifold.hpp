#pragma once
// ============================================================
// include/angular_manifold.hpp — Manifold Angular (Hipersuperficie)
// ============================================================
//
// REPRESENTACIÓN:
//   Hipersuperficie en R^dim definida por un centro c, 
//   un vector principal v (eje polar), y un umbral de radio R(θ):
//     R(θ) = a_0 + Σ_{k=1}^H a_k cos(kθ)
//
// CLASIFICACIÓN (signed distance):
//   r = ||x - c||_w
//   cos(θ) = <x - c, v>_w / (r * ||v||_w)
//   d(x) = r - R(θ)
//     d < 0  →  dentro del manifold (clase positiva)
//     d ≥ 0  →  fuera del manifold (clase negativa)
//
// CROMOSOMA:
//   center (dim), v (dim), w (dim), coefs (H + 1)
// ============================================================
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "utils.hpp"

namespace nsga2 {

struct AngularManifold {
    // ─── Parámetros ──────────────────────────────────────────
    int dim;
    int n_harmonics; // H

    std::vector<double> center; // c ∈ R^dim
    std::vector<double> v;      // v ∈ R^dim (vector principal)
    std::vector<double> w;      // feature weights
    std::vector<double> coefs;  // a_0, a_1, ..., a_H

    // ─── Configuración ───────────────────────────────────────
    static constexpr double AMP_MAX   = 1.0;
    static constexpr double CENTER_LO = 0.0;
    static constexpr double CENTER_HI = 1.0;

    // ─── Constructores ────────────────────────────────────────
    AngularManifold() : dim(0), n_harmonics(0) {}

    AngularManifold(int dim_, int n_harmonics_)
        : dim(dim_), n_harmonics(n_harmonics_),
          center(dim_, 0.5),
          v(dim_, 1.0 / std::sqrt(dim_)),
          w(dim_, 1.0),
          coefs(n_harmonics_ + 1, 0.0) {
        assert(dim_ > 0 && n_harmonics_ > 0);
    }

    bool valid() const {
        return dim > 0 && n_harmonics > 0
            && (int)center.size() == dim
            && (int)v.size() == dim
            && (int)w.size() == dim
            && (int)coefs.size() == n_harmonics + 1;
    }

    // Normaliza el vector principal v
    void normalize_v() {
        double norm = 0.0;
        for (int i = 0; i < dim; ++i) norm += w[i] * v[i] * v[i];
        if (norm < 1e-12) {
            for (int i = 0; i < dim; ++i) v[i] = 1.0 / std::sqrt(dim);
            return;
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < dim; ++i) v[i] /= norm;
    }

    // ─── Inferencia Matemática ───────────────────────────────

    // Evalúa R(θ) = a_0 + Σ a_k cos(kθ) usando Chebyshev de 1ra especie
    double evaluate_radius(double cos_theta) const {
        double r = coefs[0];
        if (n_harmonics >= 1) {
            double t_prev = 1.0;          // T_0(x)
            double t_curr = cos_theta;    // T_1(x)
            r += coefs[1] * t_curr;
            for (int k = 2; k <= n_harmonics; ++k) {
                double t_next = 2.0 * cos_theta * t_curr - t_prev; // T_k(x)
                r += coefs[k] * t_next;
                t_prev = t_curr;
                t_curr = t_next;
            }
        }
        return r;
    }

    double signed_distance(const std::vector<double>& x) const {
        double r2 = 0.0, v2 = 0.0, dot = 0.0;
        for (int i = 0; i < dim; ++i) {
            double ui = x[i] - center[i];
            r2  += w[i] * ui * ui;
            v2  += w[i] * v[i] * v[i];
            dot += w[i] * ui * v[i];
        }

        double r = std::sqrt(r2);
        if (r < 1e-12 || v2 < 1e-12) {
            return r - coefs[0]; // En el centro, cos no está bien definido, usamos r - a_0
        }

        double cos_theta = dot / (r * std::sqrt(v2));
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);

        double R_theta = evaluate_radius(cos_theta);
        return r - R_theta;
    }

    bool is_inside(const std::vector<double>& x) const {
        return signed_distance(x) < 0.0;
    }

    // ─── Medidas de complejidad ───────────────────────────────
    
    int complexity() const { return n_harmonics; }

    double coef_energy() const {
        double e = 0;
        for (auto val : coefs) e += std::abs(val);
        return e;
    }
    double arc_length(int = 0) const {
        return coef_energy(); 
    }

    // ─── Operaciones sobre el cromosoma ──────────────────────

    void add_harmonic() {
        ++n_harmonics;
        coefs.push_back(rand_double(-AMP_MAX, AMP_MAX));
    }

    bool remove_harmonic() {
        if (n_harmonics <= 1) return false;
        coefs.pop_back();
        --n_harmonics;
        return true;
    }

    void randomize(double scale = AMP_MAX) {
        for (int i = 0; i < dim; ++i) {
            center[i] = rand_double(CENTER_LO, CENTER_HI);
            v[i]      = rand_normal(0.0, 1.0);
        }
        normalize_v();
        for (auto& c : coefs)
            c = rand_double(-scale, scale);
        // Queremos que el radio base (a_0) sea positivo generalmente
        coefs[0] = std::abs(coefs[0]) + 0.1;
    }

    void set_center_near(const std::vector<double>& c, double noise = 0.05) {
        assert((int)c.size() == dim);
        for (int i = 0; i < dim; ++i)
            center[i] = c[i] + rand_normal(0.0, noise);
    }

    int n_params() const { return 3 * dim + (int)coefs.size(); }
};

} // namespace nsga2
