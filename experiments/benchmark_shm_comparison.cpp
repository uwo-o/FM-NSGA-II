// ============================================================
// experiments/benchmark_shm_comparison.cpp — Comparativa 4 Kernels NSGA-II
// FM vs AM vs GAM vs SHM
// ============================================================
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include "dataset.hpp"
#include "nsga2.hpp"
#include "angular_manifold.hpp"
#include "gam_manifold.hpp"
#include "sh_manifold.hpp"
#include "metrics.hpp"

using namespace nsga2;

struct ModelResult {
    std::string name;
    double acc;
    double f1;
    double time_ms;
};

template <typename ManifoldType>
ModelResult train_and_eval(const std::string& name, const Dataset& train, const Dataset& test,
                           int min_h, int max_h, int gens=100) {
    NSGAConfig cfg;
    cfg.pop_size = 60;
    cfg.max_gen = gens;
    cfg.patience = 30;
    cfg.min_harmonics = min_h;
    cfg.max_harmonics = max_h;
    cfg.verbose = false;
    cfg.obj_cfg.lambda = 0.01;
    cfg.obj_cfg.lambda_l1 = 0.05;

    auto t0 = std::chrono::high_resolution_clock::now();
    Classifier<ManifoldType> clf(cfg);
    clf.fit(train);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    auto preds = clf.predict_all(test);
    std::vector<int> truth;
    for (auto& s : test.samples) truth.push_back(s.label);

    return {
        name,
        multiclass_accuracy(truth, preds),
        macro_f1(truth, preds, test.class_labels),
        ms
    };
}

int main() {
    std::vector<std::pair<std::string, std::string>> datasets = {
        {"Iris (4D, 3-class)", "data/iris.csv"},
        {"Wine (13D, 3-class)", "data/wine.csv"},
        {"Breast Cancer (30D, 2-class)", "data/breast_cancer.csv"},
        {"Adversarial (52D, 2-class)", "data/adversarial_52d.csv"},
        {"Blobs 2D", "data/blobs_2d.csv"},
        {"Two Moons 2D", "data/moons_2d.csv"},
        {"Circles 2D", "data/circles_2d.csv"}
    };

    std::cout << "========================================================================================================\n";
    std::cout << "               COMPARATIVA DE LOS 4 KERNELS TOPOLÓGICOS (NSGA-II)\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(30) << "Dataset"
              << " | " << std::setw(15) << "FM-NSGA-II"
              << " | " << std::setw(15) << "AM-NSGA-II"
              << " | " << std::setw(15) << "GAM-NSGA-II"
              << " | " << std::setw(15) << "SHM-NSGA-II (Nuevo)" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& [label, path] : datasets) {
        Dataset data;
        try {
            data.load_csv(path);
            data.normalize();
        } catch (...) {
            continue;
        }

        auto [train, test] = data.train_test_split(0.3, 42);

        auto r_fm  = train_and_eval<FourierManifold>("FM", train, test, 1, 8, 100);
        auto r_am  = train_and_eval<AngularManifold>("AM", train, test, 1, 8, 100);
        auto r_gam = train_and_eval<GAMManifold>("GAM", train, test, 1, 4, 100);
        auto r_shm = train_and_eval<SphericalHarmonicManifold>("SHM", train, test, 0, 4, 100);

        auto fmt_res = [](const ModelResult& r) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << (r.acc * 100.0) << "% (" << (int)r.time_ms << "ms)";
            return ss.str();
        };

        std::cout << std::left << std::setw(30) << label
                  << " | " << std::setw(15) << fmt_res(r_fm)
                  << " | " << std::setw(15) << fmt_res(r_am)
                  << " | " << std::setw(15) << fmt_res(r_gam)
                  << " | " << std::setw(15) << fmt_res(r_shm) << "\n";
    }
    std::cout << "========================================================================================================\n";

    return 0;
}
