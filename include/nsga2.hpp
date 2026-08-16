#pragma once
// ============================================================
// include/nsga2.hpp — Motor NSGA-II (Deb et al., 2002)
// ============================================================
//
// ALGORITMO:
//   1. Inicializar población P (size N)
//   2. Evaluar objetivos
//   3. Loop max_gen veces:
//      a. Crear offspring Q (N individuos) via torneo+cruzamiento+mutación
//      b. R = P ∪ Q (2N individuos)
//      c. Fast non-dominated sort → frentes F1, F2, ...
//      d. Crowding distance en cada frente
//      e. Nueva P: llenar con frentes hasta completar N individuos
//         (último frente parcial: ordenar por crowding desc)
//   4. Retornar frente de Pareto (F1 de la última generación)
//
// MULTI-CLASE: ver FMClassifier (one-vs-rest con K llamadas)
// ============================================================
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <numeric>
#include <mutex>
#ifdef _OPENMP
#  include <omp.h>
#endif
#include "manifold.hpp"
#include "objectives.hpp"
#include "operators.hpp"

namespace nsga2 {

// ─── Individual ──────────────────────────────────────────────
struct Individual {
    FourierManifold manifold;
    std::array<double, 2> obj = {1.0, 1.0}; // [error, complexity]
    int    rank     = 0;
    double crowding = 0.0;

    // Dominancia estricta (minimización de ambos objetivos)
    bool dominates(const Individual& o) const {
        return (obj[0] <= o.obj[0] && obj[1] <= o.obj[1]) &&
               (obj[0] <  o.obj[0] || obj[1] <  o.obj[1]);
    }
};

// ─── Configuración ───────────────────────────────────────────
struct NSGAConfig {
    int pop_size      = 60;    // reducido: 100 → 60
    int max_gen       = 150;   // reducido: 300 → 150
    int patience      = 40;    // early stopping: gens sin mejora
    double batch_ratio = 1.0;  // 1.0 = usar todo el dataset, <1.0 mini-batching
    double p_cross    = 0.9;
    int min_harmonics = 1;
    int max_harmonics = 8;
    OperatorConfig op_cfg;
    ObjectiveConfig obj_cfg;
    bool verbose      = true;
    int  log_every    = 50;
};

// ─── Motor NSGA-II ────────────────────────────────────────────
class NSGA2 {
public:
    NSGA2(const NSGAConfig& cfg, ObjFn obj_fn, int dim, int dataset_size,
          std::vector<double> center_hint = {})
        : cfg_(cfg), obj_fn_(obj_fn), dim_(dim), dataset_size_(dataset_size),
          center_hint_(std::move(center_hint)) {}

    // Ejecuta el algoritmo y retorna el frente de Pareto
    std::vector<Individual> run() {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<int> empty_idx;
        std::vector<int> batch_idx;
        int batch_size = std::max(1, (int)(dataset_size_ * cfg_.batch_ratio));

        auto get_indices = [&](int gen) -> const std::vector<int>& {
            if (cfg_.batch_ratio >= 1.0 || gen >= cfg_.max_gen - 10) return empty_idx;
            batch_idx.resize(batch_size);
            for (int i = 0; i < batch_size; ++i)
                batch_idx[i] = rand_int(0, dataset_size_ - 1);
            return batch_idx;
        };

        std::vector<Individual> pop;
        init_population(pop);
        evaluate(pop, get_indices(0));
        fast_sort_and_crowd(pop);

        double best_err_prev = 1.0;
        int    stagnation    = 0;

        for (int gen = 1; gen <= cfg_.max_gen; ++gen) {
            const auto& current_indices = get_indices(gen);
            
            // Re-evaluar la población actual si los índices cambiaron
            if (cfg_.batch_ratio < 1.0 && gen < cfg_.max_gen - 10) {
                evaluate(pop, current_indices);
                fast_sort_and_crowd(pop);
            }

            auto offspring = make_offspring(pop);
            evaluate(offspring, current_indices);

            // R = P ∪ Q
            std::vector<Individual> combined = pop;
            combined.insert(combined.end(), offspring.begin(), offspring.end());

            fast_sort_and_crowd(combined);

            // Seleccionar los mejores N individuos
            pop.clear();
            pop.reserve(cfg_.pop_size);
            std::stable_sort(combined.begin(), combined.end(),
                [](const Individual& a, const Individual& b){
                    if (a.rank != b.rank) return a.rank < b.rank;
                    return a.crowding > b.crowding;
                });
            for (auto& ind : combined) {
                if ((int)pop.size() >= cfg_.pop_size) break;
                pop.push_back(ind);
            }

            // ── Logging ──────────────────────────────────────
            auto front = pareto_front(pop);
            double best_err = 1.0;
            for (auto& ind : front)
                best_err = std::min(best_err, ind.obj[0]);

            if (cfg_.verbose && gen % cfg_.log_every == 0) {
                auto t1 = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t1-t0).count();
                std::cout << "  gen=" << std::setw(4) << gen
                          << "  front_size=" << std::setw(3) << front.size()
                          << "  best_err="   << std::fixed << std::setprecision(4) << best_err
                          << "  t=" << std::fixed << std::setprecision(1) << elapsed << "s"
                          << std::defaultfloat << "\n";
            }

            // ── Early stopping ────────────────────────────────
            if (cfg_.patience > 0) {
                double current_best_err = pop.front().obj[0];
                if (cfg_.batch_ratio >= 1.0 || gen >= cfg_.max_gen - 10) {
                    if (current_best_err < best_err_prev - 1e-4) {
                        best_err_prev = current_best_err;
                        stagnation = 0;
                    } else {
                        stagnation++;
                    }
                }

                if (cfg_.patience > 0 && stagnation >= cfg_.patience) {
                    if (cfg_.verbose)
                        std::cout << "  [early stop] gen=" << gen
                                  << " (sin mejora en " << cfg_.patience << " gens)\n";
                    break;
                }
            }
        }

        pareto_ = pareto_front(pop);
        return pareto_;
    }

    // Individuo con mínimo error del frente de Pareto
    Individual best_accuracy() const {
        assert(!pareto_.empty());
        return *std::min_element(pareto_.begin(), pareto_.end(),
            [](const Individual& a, const Individual& b){ return a.obj[0] < b.obj[0]; });
    }

    // Punto de codo: mínima suma normalizada de los dos objetivos
    Individual knee_point() const {
        assert(!pareto_.empty());
        double e_max = 0, c_max = 0;
        for (auto& ind : pareto_) {
            e_max = std::max(e_max, ind.obj[0]);
            c_max = std::max(c_max, ind.obj[1]);
        }
        return *std::min_element(pareto_.begin(), pareto_.end(),
            [&](const Individual& a, const Individual& b){
                double sa = a.obj[0]/e_max + a.obj[1]/c_max;
                double sb = b.obj[0]/e_max + b.obj[1]/c_max;
                return sa < sb;
            });
    }

    const std::vector<Individual>& pareto_front_cached() const { return pareto_; }

private:
    NSGAConfig cfg_;
    ObjFn      obj_fn_;
    int        dim_;
    int        dataset_size_;
    std::vector<double>     center_hint_; // centroide de clase positiva
    std::vector<Individual> pareto_;

    // ─── Inicialización con hint de centro ────────────────────────
    void init_population(std::vector<Individual>& pop) {
        pop.resize(cfg_.pop_size);
        for (auto& ind : pop) {
            int n_harm = rand_int(cfg_.min_harmonics, cfg_.max_harmonics + 1);
            ind.manifold = FourierManifold(dim_, n_harm);
            ind.manifold.randomize();
            // Si hay hint de centro, inicializar cerca del centroide positivo
            if (!center_hint_.empty())
                ind.manifold.set_center_near(center_hint_, 0.1);
        }
    }

    // ─── Evaluación (paralela con OpenMP) ────────────────────
    void evaluate(std::vector<Individual>& pop, const std::vector<int>& indices) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < (int)pop.size(); ++i) {
            pop[i].obj = obj_fn_(pop[i].manifold, indices);
        }
    }

    // ─── Fast Non-Dominated Sort ──────────────────────────────
    // Asigna rank y crowding a todos los individuos en-lugar
    void fast_sort_and_crowd(std::vector<Individual>& pop) {
        int n = (int)pop.size();
        std::vector<int> dom_count(n, 0);
        std::vector<std::vector<int>> dom_set(n);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (pop[i].dominates(pop[j]))      dom_set[i].push_back(j);
                else if (pop[j].dominates(pop[i])) ++dom_count[i];
            }

        // Reservar espacio suficiente en fronts para evitar realloc
        std::vector<std::vector<int>> fronts;
        fronts.reserve(n + 1);  // <-- FIX: evita invalidación por realloc
        fronts.emplace_back();

        for (int i = 0; i < n; ++i)
            if (dom_count[i] == 0) { pop[i].rank = 1; fronts[0].push_back(i); }

        int fi = 0;
        while (fi < (int)fronts.size() && !fronts[fi].empty()) {
            std::vector<int> next;
            // Copiar el frente actual antes de push_back para evitar
            // que la referencia se invalide si fronts se reasigna
            const std::vector<int> current = fronts[fi];
            for (int i : current)
                for (int j : dom_set[i]) {
                    if (--dom_count[j] == 0) {
                        pop[j].rank = fi + 2;
                        next.push_back(j);
                    }
                }
            if (!next.empty()) fronts.push_back(std::move(next));
            ++fi;
        }

        // Crowding distance por frente
        for (auto& fr : fronts)
            assign_crowding(pop, fr);

    }

    void assign_crowding(std::vector<Individual>& pop, const std::vector<int>& front) {
        int sz = (int)front.size();
        if (sz <= 2) {
            for (int i : front) pop[i].crowding = std::numeric_limits<double>::max();
            return;
        }
        for (int i : front) pop[i].crowding = 0.0;

        for (int m = 0; m < 2; ++m) {
            std::vector<int> sorted = front;
            std::sort(sorted.begin(), sorted.end(),
                [&](int a, int b){ return pop[a].obj[m] < pop[b].obj[m]; });
            pop[sorted.front()].crowding = std::numeric_limits<double>::max();
            pop[sorted.back() ].crowding = std::numeric_limits<double>::max();
            double range = pop[sorted.back()].obj[m] - pop[sorted.front()].obj[m];
            if (range < 1e-12) continue;
            for (int k = 1; k < sz - 1; ++k) {
                int i = sorted[k];
                if (pop[i].crowding < 1e18)
                    pop[i].crowding += (pop[sorted[k+1]].obj[m] - pop[sorted[k-1]].obj[m]) / range;
            }
        }
    }

    // ─── Selección por torneo binario ─────────────────────────
    const Individual& tournament(const std::vector<Individual>& pop) {
        int a = rand_int(0, (int)pop.size());
        int b = rand_int(0, (int)pop.size());
        const auto& ia = pop[a]; const auto& ib = pop[b];
        if (ia.rank < ib.rank) return ia;
        if (ib.rank < ia.rank) return ib;
        return (ia.crowding >= ib.crowding) ? ia : ib;
    }

    // ─── Generación de offspring ──────────────────────────────
    std::vector<Individual> make_offspring(const std::vector<Individual>& pop) {
        std::vector<Individual> offspring;
        offspring.reserve(cfg_.pop_size);

        while ((int)offspring.size() < cfg_.pop_size) {
            const auto& p1 = tournament(pop);
            const auto& p2 = tournament(pop);

            FourierManifold m1, m2;
            if (rand_bool(cfg_.p_cross)) {
                auto [c1, c2] = crossover(p1.manifold, p2.manifold, cfg_.op_cfg);
                m1 = std::move(c1);
                m2 = std::move(c2);
            } else {
                m1 = p1.manifold;
                m2 = p2.manifold;
            }

            // Asegurar límites de harmónicos
            while (m1.n_harmonics > cfg_.max_harmonics) m1.remove_harmonic();
            while (m1.n_harmonics < cfg_.min_harmonics) m1.add_harmonic();
            while (m2.n_harmonics > cfg_.max_harmonics) m2.remove_harmonic();
            while (m2.n_harmonics < cfg_.min_harmonics) m2.add_harmonic();

            offspring.push_back({mutate(m1, cfg_.op_cfg), {1.0, 1.0}, 0, 0.0});
            if ((int)offspring.size() < cfg_.pop_size)
                offspring.push_back({mutate(m2, cfg_.op_cfg), {1.0, 1.0}, 0, 0.0});
        }
        return offspring;
    }

    // ─── Frente de Pareto de la población ─────────────────────
    std::vector<Individual> pareto_front(const std::vector<Individual>& pop) {
        std::vector<Individual> front;
        for (auto& ind : pop)
            if (ind.rank == 1) front.push_back(ind);
        return front;
    }
};

// ============================================================
// FMClassifier — Wrapper multi-clase (one-vs-rest)
// ============================================================
class FMClassifier {
public:
    FMClassifier(NSGAConfig cfg = {}) : cfg_(cfg) {}

    // Entrenamiento: corre NSGA-II por cada clase (one-vs-rest)
    // Las clases se procesan en paralelo con OpenMP.
    void fit(const Dataset& data) {
        class_labels_ = data.class_labels;
        n_classes_     = data.n_classes;
        dim_           = data.n_features;

        int K = n_classes_;
        manifolds_.resize(K);

        // Vectores de trabajo por clase (precalcular fuera del parallel)
        std::vector<Dataset>          binaries(K);
        std::vector<std::vector<double>> hints(K);
        for (int ki = 0; ki < K; ++ki) {
            int cls = class_labels_[ki];
            binaries[ki] = data.as_binary(cls);
            std::vector<std::vector<double>> pos_pts;
            for (auto& s : binaries[ki].samples)
                if (s.label == 1) pos_pts.push_back(s.x);
            hints[ki] = pos_pts.empty()
                        ? std::vector<double>(dim_, 0.5)
                        : centroid(pos_pts);
        }

        // mutex solo para output de consola
        std::mutex cout_mtx;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int ki = 0; ki < K; ++ki) {
            int cls = class_labels_[ki];
            if (cfg_.verbose) {
                std::lock_guard<std::mutex> lk(cout_mtx);
                std::cout << "\n[NSGA-II] Clase " << cls << " vs. resto...\n";
            }

            auto obj_fn = make_objective(binaries[ki], cfg_.obj_cfg);
            NSGA2 engine(cfg_, obj_fn, dim_, binaries[ki].size(), hints[ki]);
            auto front = engine.run();
            auto best  = engine.best_accuracy();
            manifolds_[ki] = best.manifold;

            if (cfg_.verbose) {
                std::lock_guard<std::mutex> lk(cout_mtx);
                std::cout << "  [Clase " << cls << "] → error="
                          << std::fixed << std::setprecision(4) << best.obj[0]
                          << "  complexity=" << std::setprecision(4) << best.obj[1]
                          << "  harmonics="  << best.manifold.n_harmonics
                          << std::defaultfloat << "\n";
            }
        }
    }

    // Predicción: clase con mínima distancia firmada (más "dentro")
    int predict(const std::vector<double>& x) const {
        double best_d = std::numeric_limits<double>::max();
        int    best_c = class_labels_[0];
        for (int k = 0; k < n_classes_; ++k) {
            double d = manifolds_[k].signed_distance(x);
            if (d < best_d) { best_d = d; best_c = class_labels_[k]; }
        }
        return best_c;
    }

    // Predicción sobre dataset completo
    std::vector<int> predict_all(const Dataset& data) const {
        std::vector<int> preds;
        preds.reserve(data.size());
        for (auto& s : data.samples)
            preds.push_back(predict(s.x));
        return preds;
    }

    const std::vector<FourierManifold>& manifolds() const { return manifolds_; }
    const std::vector<int>& labels()                const { return class_labels_; }

private:
    NSGAConfig cfg_;
    int dim_ = 0, n_classes_ = 0;
    std::vector<int>             class_labels_;
    std::vector<FourierManifold> manifolds_;
};

} // namespace nsga2
