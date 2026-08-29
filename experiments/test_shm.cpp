// ============================================================
// experiments/test_shm.cpp — Test de Spherical Harmonic Manifold
// ============================================================
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "dataset.hpp"
#include "nsga2.hpp"
#include "sh_manifold.hpp"
#include "metrics.hpp"

using namespace nsga2;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Spherical Harmonic Manifold (SH-NSGA-II) Benchmark\n";
    std::cout << "========================================================\n\n";

    std::vector<std::string> datasets = {
        "data/moons_2d.csv",
        "data/circles_2d.csv",
        "data/spiral_2d.csv",
        "data/iris.csv",
        "data/wine.csv",
        "data/breast_cancer.csv"
    };

    for (const auto& path : datasets) {
        Dataset data;
        try {
            data.load_csv(path);
            data.normalize();
        } catch (const std::exception& e) {
            std::cerr << "Error cargando " << path << ": " << e.what() << "\n";
            continue;
        }

        auto [train, test] = data.train_test_split(0.3, 42);

        NSGAConfig cfg;
        cfg.pop_size = 60;
        cfg.max_gen = 100;
        cfg.patience = 30;
        cfg.min_harmonics = 0; // L=0 (esfera)
        cfg.max_harmonics = 4; // L=4 ((4+1)^2 = 25 coefs SH)
        cfg.verbose = false;
        cfg.obj_cfg.lambda = 0.01;
        cfg.obj_cfg.lambda_l1 = 0.05;

        auto t0 = std::chrono::high_resolution_clock::now();
        Classifier<SphericalHarmonicManifold> clf(cfg);
        clf.fit(train);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        auto preds = clf.predict_all(test);
        std::vector<int> truth;
        for (auto& s : test.samples) truth.push_back(s.label);

        double acc = multiclass_accuracy(truth, preds);
        double f1  = macro_f1(truth, preds, test.class_labels);

        std::cout << std::left << std::setw(30) << path
                  << " | Dim: " << std::setw(3) << train.n_features
                  << " | Clases: " << std::setw(2) << train.n_classes
                  << " | Acc: " << std::fixed << std::setprecision(4) << acc
                  << " | F1: " << f1
                  << " | Time: " << std::setprecision(1) << ms << " ms\n";
    }

    return 0;
}
