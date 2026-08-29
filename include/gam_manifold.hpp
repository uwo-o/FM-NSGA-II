#pragma once
// ============================================================
// include/gam_manifold.hpp — GAM Manifold Kernel
// ============================================================
//
// REPRESENTACIÓN:
//   Hipersuperficie implícita definida por un Generalized
//   Additive Model (GAM):
//
//     d(x) = β₀ + Σ_{i=1}^{D} w_i · f_i(x_i)
//
//   donde cada f_i es una serie de Fourier truncada sobre [0,1]:
//
//     f_i(x) = Σ_{k=1}^{H} [ a_{ik}·cos(kπx) + b_{ik}·sin(kπx) ]
//
//   La base cos/sin sobre [0,1] (con periodo 2 en lugar de 2π)
//   garantiza que f_i es suave y periódica.
//
// CLASIFICACIÓN (distancia firmada):
//   d < 0  →  dentro (clase positiva)
//   d ≥ 0  →  fuera  (clase negativa)
//
//   NOTA: a diferencia de FM/AM no hay garantía topológica de
//   región compacta. La complejidad de O2 se mide como energía
//   total de los coeficientes + número de armónicos activos.
//
// CROMOSOMA (flat):
//   [ β₀,
//     w_0, a_{0,1}, b_{0,1}, ..., a_{0,H}, b_{0,H},
//     w_1, a_{1,1}, b_{1,1}, ..., a_{1,H}, b_{1,H},
//     ...
//     w_{D-1}, a_{D-1,1}, b_{D-1,1}, ..., a_{D-1,H}, b_{D-1,H} ]
//
//   Tamaño: 1 + D * (1 + 2*H)
//
// VENTAJA SOBRE FM / AM:
//   - Interpretabilidad perfecta: f_i(x_i) es la contribución
//     de la feature i a la decisión.
//   - Costo de inferencia O(H*D) — sin búsqueda de t*.
//   - Feature selection implícito: w_i → 0 excluye la feature.
//
// LIMITACIÓN:
//   - No modela interacciones x_i × x_j directamente.
//   - Para datasets con estructura no-aditiva (Two-Moons)
//     puede requerir más armónicos que FM/AM.
// ============================================================
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "utils.hpp"

namespace nsga2 {

struct GAMManifold {
    // ─── Parámetros ──────────────────────────────────────────
    int dim;          // dimensión del espacio ambiente D
    int n_harmonics;  // H (armónicos de Fourier por feature)

    double              beta0;   // intercepto global β₀
    std::vector<double> w;       // pesos de feature [dim]   ∈ [0,1]
    std::vector<double> coefs;   // coeficientes Fourier [dim * 2 * H]
                                 // layout: [ a_{i,1},b_{i,1}, ..., a_{i,H},b_{i,H} ]
                                 //         para i = 0..dim-1 (bloques de 2H)

    // ─── Configuración ───────────────────────────────────────
    static constexpr double AMP_MAX    = 1.0;
    static constexpr double BETA_MAX   = 1.0;

    // ─── Constructores ────────────────────────────────────────
    GAMManifold() : dim(0), n_harmonics(0), beta0(0.0) {}

    GAMManifold(int dim_, int n_harmonics_)
        : dim(dim_), n_harmonics(n_harmonics_),
          beta0(0.0),
          w(dim_, 1.0),
          coefs(dim_ * 2 * n_harmonics_, 0.0) {
        assert(dim_ > 0 && n_harmonics_ > 0);
    }

    bool valid() const {
        return dim > 0 && n_harmonics > 0
            && (int)w.size()     == dim
            && (int)coefs.size() == dim * 2 * n_harmonics;
    }

    // ─── Acceso a coeficientes ────────────────────────────────
    // Feature i (0-based), armónico k (1-based)
    double& a(int i, int k)       { return coefs[i * 2 * n_harmonics + (k-1)*2    ]; }
    double& b(int i, int k)       { return coefs[i * 2 * n_harmonics + (k-1)*2 + 1]; }
    const double& a(int i, int k) const { return coefs[i * 2 * n_harmonics + (k-1)*2    ]; }
    const double& b(int i, int k) const { return coefs[i * 2 * n_harmonics + (k-1)*2 + 1]; }

    // ─── Evaluación de la función por feature ─────────────────
    // f_i(x_i) = Σ_{k=1}^H [ a_{ik}·cos(kπx_i) + b_{ik}·sin(kπx_i) ]
    double eval_fi(int i, double xi) const {
        double v = 0.0;
        for (int k = 1; k <= n_harmonics; ++k) {
            const double kpix = k * M_PI * xi;
            v += a(i,k) * std::cos(kpix) + b(i,k) * std::sin(kpix);
        }
        return v;
    }

    // ─── Distancia firmada ────────────────────────────────────
    // d(x) = β₀ + Σ_i w_i · f_i(x_i)
    // < 0 → dentro (clase positiva)
    // ≥ 0 → fuera  (clase negativa)
    double signed_distance(const std::vector<double>& x) const {
        assert((int)x.size() == dim);
        double d = beta0;
        for (int i = 0; i < dim; ++i)
            d += w[i] * eval_fi(i, x[i]);
        return d;
    }

    bool is_inside(const std::vector<double>& x) const {
        return signed_distance(x) < 0.0;
    }

    // ─── Contribuciones por feature (interpretabilidad) ───────
    // Retorna el aporte w_i * f_i(x_i) de cada feature
    std::vector<double> feature_contributions(const std::vector<double>& x) const {
        std::vector<double> contrib(dim);
        for (int i = 0; i < dim; ++i)
            contrib[i] = w[i] * eval_fi(i, x[i]);
        return contrib;
    }

    // ─── Medidas de complejidad ───────────────────────────────

    int complexity() const { return n_harmonics; }

    // Energía L2 total de los coeficientes (penalización de oscilación)
    double coef_energy() const {
        double e = 0.0;
        for (auto v : coefs) e += v * v;
        return e;
    }

    // arc_length(): para compatibilidad con objectives.hpp
    // Aquí lo aproximamos como la energía de las derivadas (roughness):
    //   Σ_i Σ_k k² (a²_{ik} + b²_{ik})
    // Penaliza armónicos de alta frecuencia más severamente.
    double arc_length(int /*n_samples*/ = 0) const {
        double roughness = 0.0;
        for (int i = 0; i < dim; ++i)
            for (int k = 1; k <= n_harmonics; ++k)
                roughness += static_cast<double>(k * k) *
                             (a(i,k)*a(i,k) + b(i,k)*b(i,k));
        return roughness;
    }

    // ─── Operaciones sobre el cromosoma ──────────────────────

    void add_harmonic() {
        ++n_harmonics;
        // Añade a_{i,H} y b_{i,H} para cada feature i
        // Insertamos al final del bloque de cada feature (requiere reorganizar)
        std::vector<double> new_coefs(dim * 2 * n_harmonics, 0.0);
        for (int i = 0; i < dim; ++i) {
            // Copiar los 2*(H-1) coeficientes anteriores de la feature i
            int old_block = i * 2 * (n_harmonics - 1);
            int new_block = i * 2 * n_harmonics;
            for (int j = 0; j < 2 * (n_harmonics - 1); ++j)
                new_coefs[new_block + j] = coefs[old_block + j];
            // Nuevos coeficientes: ERC aleatorio
            new_coefs[new_block + 2*(n_harmonics-1)    ] = rand_double(-AMP_MAX, AMP_MAX);
            new_coefs[new_block + 2*(n_harmonics-1) + 1] = rand_double(-AMP_MAX, AMP_MAX);
        }
        coefs = std::move(new_coefs);
    }

    bool remove_harmonic() {
        if (n_harmonics <= 1) return false;
        std::vector<double> new_coefs(dim * 2 * (n_harmonics - 1));
        for (int i = 0; i < dim; ++i) {
            int old_block = i * 2 * n_harmonics;
            int new_block = i * 2 * (n_harmonics - 1);
            for (int j = 0; j < 2 * (n_harmonics - 1); ++j)
                new_coefs[new_block + j] = coefs[old_block + j];
        }
        coefs = std::move(new_coefs);
        --n_harmonics;
        return true;
    }

    // Inicialización aleatoria completa (ERC)
    void randomize(double scale = AMP_MAX) {
        beta0 = rand_double(-BETA_MAX, BETA_MAX);
        for (auto& wi : w)
            wi = rand_double(0.0, 1.0);
        for (auto& c : coefs)
            c = rand_double(-scale, scale);
    }

    // Fija beta0 para que d(centroide_clase) ≈ -0.5
    // (heurística de inicialización para One-vs-Rest)
    void set_bias_from_centroid(const std::vector<double>& c) {
        // Evalúa el GAM en el centroide sin bias
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
            sum += w[i] * eval_fi(i, c[i]);
        // Queremos beta0 + sum ≈ -0.5 → beta0 = -0.5 - sum
        beta0 = -0.5 - sum;
    }

    // Alias requerido por nsga2.hpp::init_population — equivalente a set_bias_from_centroid
    // con ruido Gaussiano sobre beta0 para diversidad inicial.
    void set_center_near(const std::vector<double>& c, double noise = 0.05) {
        assert((int)c.size() == dim);
        set_bias_from_centroid(c);
        beta0 += rand_normal(0.0, noise);
    }

    // Número total de parámetros del cromosoma
    int n_params() const { return 1 + dim + (int)coefs.size(); }
};

} // namespace nsga2
