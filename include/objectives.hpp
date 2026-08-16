#pragma once
// ============================================================
// include/objectives.hpp — Funciones objetivo para NSGA-II
// ============================================================
//
// Objetivo 1 (minimizar): tasa de error de clasificación
//   error = (# mal clasificados) / total
//
// Objetivo 2 (minimizar): complejidad del manifold
//   complexity = λ * (arc_length / arc_baseline) + (1-λ) * (N / N_max)
//   donde N = número de harmónicos y N_max = máximo permitido
// ============================================================
#include <array>
#include <vector>
#include <functional>
#include "manifold.hpp"
#include "dataset.hpp"

namespace nsga2 {

// ─── Función objetivo compuesta ──────────────────────────────
// Retorna [error_rate ∈ [0,1], complexity ∈ [0,1]]
struct ObjectiveConfig {
    int    max_harmonics  = 8;
    double arc_baseline   = 10.0; // longitud de arco de referencia (espacio normalizado)
    double lambda         = 0.5;  // peso entre arco y conteo de harmónicos
};

using ObjFn = std::function<std::array<double, 2>(const FourierManifold&, const std::vector<int>&)>;

inline std::array<double, 2> evaluate_objectives(
        const FourierManifold& m,
        const Dataset& binary_data,
        const std::vector<int>& indices,
        const ObjectiveConfig& cfg = {}) {

    // ── Objetivo 1: error de clasificación ───────────────────
    int n_wrong = 0;
    int total = indices.empty() ? binary_data.size() : indices.size();
    if (total == 0) return {1.0, 1.0};

    if (indices.empty()) {
        for (auto& s : binary_data.samples) {
            bool pred = m.is_inside(s.x);
            bool truth = (s.label == 1);
            if (pred != truth) ++n_wrong;
        }
    } else {
        for (int idx : indices) {
            const auto& s = binary_data.samples[idx];
            bool pred = m.is_inside(s.x);
            bool truth = (s.label == 1);
            if (pred != truth) ++n_wrong;
        }
    }
    double error = static_cast<double>(n_wrong) / total;

    // ── Objetivo 2: complejidad del manifold ─────────────────
    double arc_norm  = m.arc_length(100) / cfg.arc_baseline;
    double harm_norm = static_cast<double>(m.n_harmonics) / cfg.max_harmonics;
    double complexity = cfg.lambda * std::min(arc_norm, 1.0)
                      + (1.0 - cfg.lambda) * std::min(harm_norm, 1.0);

    return { error, std::min(complexity, 1.0) };
}

inline ObjFn make_objective(const Dataset& binary_data,
                             const ObjectiveConfig& cfg = {}) {
    return [&binary_data, cfg](const FourierManifold& m, const std::vector<int>& indices) {
        return evaluate_objectives(m, binary_data, indices, cfg);
    };
}

} // namespace nsga2
