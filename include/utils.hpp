#pragma once
// ============================================================
// include/utils.hpp — Utilidades matemáticas y RNG
// ============================================================
#include <random>
#include <thread>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <functional>
#include <cassert>

namespace nsga2 {

// ─── RNG thread-local (seguro para OpenMP) ───────────────────
// Cada hilo tiene su propio mt19937 con semilla única basada en
// thread id + tiempo, evitando race conditions y mutex overhead.
inline std::mt19937& rng() {
    thread_local std::mt19937 gen(
        std::random_device{}() ^
        static_cast<unsigned>(
            std::hash<std::thread::id>{}(std::this_thread::get_id())
        )
    );
    return gen;
}

inline void set_seed(unsigned s) {
    // Semilla base + offset por hilo para reproducibilidad parcial
    thread_local bool seeded = false;
    (void)seeded;
    rng().seed(s ^ static_cast<unsigned>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())
    ));
}

inline double rand_double(double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng());
}

inline int rand_int(int lo, int hi_excl) {
    if (lo >= hi_excl - 1) return lo;
    return std::uniform_int_distribution<int>(lo, hi_excl - 1)(rng());
}

inline bool rand_bool(double p = 0.5) {
    return std::bernoulli_distribution(p)(rng());
}

inline double rand_normal(double mean = 0.0, double sigma = 1.0) {
    return std::normal_distribution<double>(mean, sigma)(rng());
}

// ─── Búsqueda de sección áurea ───────────────────────────────
// Minimiza f en [a, b]
inline double golden_section(std::function<double(double)> f,
                              double a, double b, int iters = 40) {
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = f(c), fd = f(d);
    for (int i = 0; i < iters; ++i) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a); fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a); fd = f(d);
        }
    }
    return (a + b) / 2.0;
}

// ─── Álgebra vectorial ────────────────────────────────────────
inline double norm2_sq(const std::vector<double>& v) {
    double s = 0; for (auto x : v) s += x * x; return s;
}
inline double norm2(const std::vector<double>& v) {
    return std::sqrt(norm2_sq(v));
}
inline double dist2_sq(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i]-b[i])*(a[i]-b[i]);
    return s;
}
inline double dist2_sq(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& w) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += w[i] * (a[i]-b[i])*(a[i]-b[i]);
    return s;
}
inline double dist2(const std::vector<double>& a, const std::vector<double>& b) {
    return std::sqrt(dist2_sq(a, b));
}
inline double dist2(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& w) {
    return std::sqrt(dist2_sq(a, b, w));
}
inline std::vector<double> vsub(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
inline std::vector<double> vadd(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}
inline std::vector<double> vscale(const std::vector<double>& a, double s) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}
inline std::vector<double> centroid(const std::vector<std::vector<double>>& pts) {
    if (pts.empty()) return {};
    std::vector<double> c(pts[0].size(), 0.0);
    for (auto& p : pts) for (size_t i = 0; i < p.size(); ++i) c[i] += p[i];
    for (auto& v : c) v /= pts.size();
    return c;
}

} // namespace nsga2
