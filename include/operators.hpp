#pragma once
// ============================================================
// include/operators.hpp — Mutación ERC y cruzamiento GP-style
// ============================================================
//
// MUTACIÓN (ERC entre 0 y 2π para el dominio angular):
//   - Cada coeficiente Fourier se muta con probabilidad p_coef
//     → Reemplazo completo: nuevo ERC de U(-AMP_MAX, AMP_MAX)
//   - Con prob p_add: agregar un nuevo harmónico (ERC fresco)
//   - Con prob p_del: eliminar el último harmónico
//   - El centro muta con ruido Gaussiano
//
// CRUZAMIENTO (longitud variable, estilo GP):
//   - Alinear cromosomas por índice de harmónico (k=1, k=2, ...)
//   - Hasta min(N1, N2): para cada harmónico k, intercambiar
//     los 2*dim coeficientes {a_{ik}, b_{ik}} con prob 0.5
//   - Harmónicos extra del padre más largo: heredar con prob 0.5
//   - El centro se interpola aleatoriamente entre los dos padres
// ============================================================
#include <cassert>
#include "manifold.hpp"

namespace nsga2 {

struct OperatorConfig {
    double p_coef    = 0.1;  // prob de mutar cada coeficiente
    double p_add     = 0.05; // prob de agregar harmónico
    double p_del     = 0.05; // prob de eliminar harmónico
    double center_sigma = 0.05; // σ de la mutación del centro
    int    min_harmonics = 1;
    int    max_harmonics = 8;
};

// ─── Mutación ERC ────────────────────────────────────────────
inline FourierManifold mutate(FourierManifold m, const OperatorConfig& cfg = {}) {
    assert(m.valid());

    // 1) Mutar coeficientes Fourier (ERC)
    for (auto& c : m.coefs) {
        if (rand_bool(cfg.p_coef))
            c = rand_double(-FourierManifold::AMP_MAX, FourierManifold::AMP_MAX);
    }

    // 2) Mutar centro con ruido Gaussiano
    for (auto& ci : m.center)
        if (rand_bool(cfg.p_coef))
            ci = std::clamp(ci + rand_normal(0.0, cfg.center_sigma), 0.0, 1.0);

    // 2.5) Mutar pesos (Feature Weights)
    for (auto& wi : m.w) {
        if (rand_bool(cfg.p_coef)) {
            // Fuerte presión exploratoria o refinamiento local
            if (rand_bool(0.2)) wi = rand_double(0.0, 1.0); // Random reset
            else wi = std::clamp(wi + rand_normal(0.0, 0.2), 0.0, 1.0); // Gauss
        }
    }

    // 3) Agregar harmónico (si no supera el máximo)
    if (m.n_harmonics < cfg.max_harmonics && rand_bool(cfg.p_add))
        m.add_harmonic();

    // 4) Eliminar harmónico (si no baja del mínimo)
    if (m.n_harmonics > cfg.min_harmonics && rand_bool(cfg.p_del))
        m.remove_harmonic();

    return m;
}

// ─── Cruzamiento de longitud variable (GP-style) ─────────────
// Retorna dos hijos. Estrategia:
//   - Para harmónicos 1..min(N1,N2): intercambiar bloque de 2*dim
//     coeficientes con prob 0.5
//   - Para harmónicos extra de N_max: heredar con prob 0.5
//   - El hijo resultante tiene entre N_min y N_max harmónicos
inline std::pair<FourierManifold, FourierManifold>
crossover(const FourierManifold& p1, const FourierManifold& p2,
          const OperatorConfig& /*cfg*/ = {}) {
    assert(p1.dim == p2.dim);
    int dim   = p1.dim;
    int N_min = std::min(p1.n_harmonics, p2.n_harmonics);
    int N_max = std::max(p1.n_harmonics, p2.n_harmonics);

    // Padres en orden: longer tiene N_max, shorter tiene N_min
    const FourierManifold& longer  = (p1.n_harmonics >= p2.n_harmonics) ? p1 : p2;
    const FourierManifold& shorter = (p1.n_harmonics >= p2.n_harmonics) ? p2 : p1;

    // ── Construir coefs de hijos bloque a bloque ──────────────
    // Cada bloque = 2*dim valores para un harmónico k
    // Hijo A: inicialmente copia de longer; Hijo B: copia de shorter
    std::vector<double> coefs_a, coefs_b;
    coefs_a.reserve(2 * N_max * dim);
    coefs_b.reserve(2 * N_max * dim);

    // Harmónicos compartidos: intercambiar con prob 0.5
    for (int k = 0; k < N_min; ++k) {
        int base = k * 2 * dim;
        bool swap = rand_bool(0.5);
        for (int j = 0; j < 2 * dim; ++j) {
            coefs_a.push_back(swap ? shorter.coefs[base+j] : longer.coefs[base+j]);
            coefs_b.push_back(swap ? longer.coefs[base+j]  : shorter.coefs[base+j]);
        }
    }

    // Harmónicos extra (solo en longer): heredar con prob 0.5 a cada hijo
    int n_harm_a = N_min, n_harm_b = N_min;
    for (int k = N_min; k < N_max; ++k) {
        int base = k * 2 * dim;
        if (rand_bool(0.5)) {
            for (int j = 0; j < 2 * dim; ++j)
                coefs_a.push_back(longer.coefs[base+j]);
            ++n_harm_a;
        }
        if (rand_bool(0.5)) {
            for (int j = 0; j < 2 * dim; ++j)
                coefs_b.push_back(longer.coefs[base+j]);
            ++n_harm_b;
        }
    }

    // Asegurar al menos 1 harmónico
    if (n_harm_a == 0) { n_harm_a = 1; for (int j=0;j<2*dim;++j) coefs_a.push_back(rand_double(-FourierManifold::AMP_MAX, FourierManifold::AMP_MAX)); }
    if (n_harm_b == 0) { n_harm_b = 1; for (int j=0;j<2*dim;++j) coefs_b.push_back(rand_double(-FourierManifold::AMP_MAX, FourierManifold::AMP_MAX)); }

    // ── Centro y Pesos: interpolación aleatoria ────────────────
    FourierManifold c1(dim, n_harm_a), c2(dim, n_harm_b);
    c1.coefs = std::move(coefs_a);
    c2.coefs = std::move(coefs_b);

    for (int i = 0; i < dim; ++i) {
        double alpha_c = rand_double(0.0, 1.0);
        c1.center[i] = alpha_c * p1.center[i] + (1.0 - alpha_c) * p2.center[i];
        c2.center[i] = (1.0 - alpha_c) * p1.center[i] + alpha_c * p2.center[i];
        
        double alpha_w = rand_double(0.0, 1.0);
        c1.w[i] = alpha_w * p1.w[i] + (1.0 - alpha_w) * p2.w[i];
        c2.w[i] = (1.0 - alpha_w) * p1.w[i] + alpha_w * p2.w[i];
    }

    assert(c1.valid());
    assert(c2.valid());
    return {c1, c2};
}

} // namespace nsga2
