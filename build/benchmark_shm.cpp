#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include "dataset.hpp"
#include "nsga2.hpp"
#include "metrics.hpp"
#include "sh_manifold.hpp"

using namespace nsga2;
using Clock = std::chrono::high_resolution_clock;

int main() {
    struct DSEntry { std::string name, path; };
    std::vector<DSEntry> datasets = {
        {"Two Moons (2D)",   "../data/moons_2d.csv"},
        {"Circles (2D)",     "../data/circles_2d.csv"},
        {"Spirals (2D)",     "../data/spirals_2d.csv"},
        {"Blobs (2D)",       "../data/blobs_2d.csv"},
        {"Chessboard (2D)",  "../data/chessboard_2d.csv"},
        {"Adversarial(52D)", "../data/adversarial_52d.csv"},
        {"Iris (4D)",        "../data/iris.csv"},
        {"Wine (13D)",       "../data/wine.csv"},
        {"BreastCancer(30D)","../data/breast_cancer.csv"},
    };

    NSGAConfig cfg;
    cfg.pop_size      = 60;
    cfg.max_gen       = 150;
    cfg.patience      = 40;
    cfg.batch_ratio   = 0.3;
    cfg.p_cross       = 0.9;
    cfg.min_harmonics = 1;
    cfg.max_harmonics = 5;
    cfg.verbose       = false;

    std::ofstream csv("benchmark_shm.csv");
    csv << "dataset,classifier,accuracy,f1,train_ms\n";

    for (auto& ds : datasets) {
        Dataset data;
        try { data.load_csv(ds.path); } catch (...) { continue; }
        data.normalize();
        auto [train, test] = data.train_test_split(0.2, 42);

        auto t0 = Clock::now();
        Classifier<SphericalHarmonicManifold> clf(cfg);
        clf.fit(train);
        double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

        auto preds = clf.predict_all(test);
        std::vector<int> truth;
        for (auto& s : test.samples) truth.push_back(s.label);

        double acc = multiclass_accuracy(truth, preds);
        double f1 = macro_f1(truth, preds, test.class_labels);
        csv << ds.name << ",SHM-NSGA-II," << acc << "," << f1 << "," << ms << "\n";
        std::cout << ds.name << " SHM: acc=" << acc << " ms=" << ms << "\n";
    }
    return 0;
}
