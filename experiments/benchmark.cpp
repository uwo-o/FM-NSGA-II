// ============================================================
// experiments/benchmark.cpp — Benchmark completo multi-dataset
// ============================================================
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <memory>
#include "dataset.hpp"
#include "nsga2.hpp"
#include "metrics.hpp"
#include "knn.hpp"
#include "naive_bayes.hpp"
#include "svm.hpp"
#include "decision_tree.hpp"

using namespace nsga2;
using Clock = std::chrono::high_resolution_clock;

// ─── Resultado de una ejecución ──────────────────────────────
struct Result {
    std::string dataset;
    std::string classifier;
    double accuracy;
    double f1;
    double train_ms;
};

// ─── Función de evaluación ───────────────────────────────────
Result evaluate(const std::string& ds_name,
                const Dataset& train, const Dataset& test,
                IClassifier& clf) {
    auto t0 = Clock::now();
    clf.fit(train);
    double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    auto preds = clf.predict_all(test);
    std::vector<int> truth;
    for (auto& s : test.samples) truth.push_back(s.label);

    return {
        ds_name, clf.name(),
        multiclass_accuracy(truth, preds),
        macro_f1(truth, preds, test.class_labels),
        ms
    };
}

// ─── Función para NSGA-II ─────────────────────────────────────
Result evaluate_nsga2(const std::string& ds_name,
                      const Dataset& train, const Dataset& test,
                      const NSGAConfig& cfg) {
    auto t0 = Clock::now();
    RMClassifier clf(cfg);
    clf.fit(train);
    double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    auto preds = clf.predict_all(test);
    std::vector<int> truth;
    for (auto& s : test.samples) truth.push_back(s.label);

    return {
        ds_name, "RM-NSGA-II",
        multiclass_accuracy(truth, preds),
        macro_f1(truth, preds, test.class_labels),
        ms
    };
}

// ─── Main ────────────────────────────────────────────────────
int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║        BENCHMARK: RM-NSGA-II vs Baselines          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ─── Datasets (ajustar paths según build dir) ─────────────
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

    // ─── Configuración NSGA-II ───────────────────────────────
    NSGAConfig cfg;
    cfg.pop_size      = 60;
    cfg.max_gen       = 150;
    cfg.patience      = 40;
    cfg.batch_ratio   = 0.3; // Activar mini-batching (30%)
    cfg.p_cross       = 0.9;
    cfg.min_harmonics = 1;
    cfg.max_harmonics = 50;
    cfg.verbose       = false;  // silencioso durante benchmark
    cfg.obj_cfg.max_harmonics = 50;
    cfg.op_cfg.p_coef    = 0.12;
    cfg.op_cfg.p_add     = 0.06;
    cfg.op_cfg.p_del     = 0.04;
    cfg.obj_cfg.lambda   = 0.5;

    // ─── Resultados ───────────────────────────────────────────
    std::vector<Result> results;
    std::string current_ds = "";

    for (auto& ds_entry : datasets) {
        Dataset data;
        try {
            data.load_csv(ds_entry.path);
        } catch (...) {
            std::cerr << "⚠ Skipping " << ds_entry.name
                      << " (no encontrado: " << ds_entry.path << ")\n";
            continue;
        }
        data.normalize();
        auto [train, test] = data.train_test_split(0.2, 42);

        std::cout << "\n── " << ds_entry.name
                  << " (n=" << data.size()
                  << ", d=" << data.n_features
                  << ", K=" << data.n_classes << ") ──\n";

        // NSGA-II
        std::cout << "  [NSGA-II]..." << std::flush;
        auto r = evaluate_nsga2(ds_entry.name, train, test, cfg);
        results.push_back(r);
        std::cout << " acc=" << std::fixed << std::setprecision(4) << r.accuracy
                  << " F1=" << r.f1
                  << " (" << (int)r.train_ms << "ms)\n";

        // Baselines
        KNN knn3(3), knn7(7);
        GaussianNB nb;
        SVM svm_rbf(1.0, 1.0, 100, "rbf");
        SVM svm_linear(1.0, 1.0, 100, "linear");
        DecisionTree dt(8);
        std::vector<IClassifier*> baselines = {&knn3, &knn7, &nb, &svm_rbf, &svm_linear, &dt};
        for (auto* bl : baselines) {
            std::cout << "  [" << bl->name() << "]..." << std::flush;
            auto rb = evaluate(ds_entry.name, train, test, *bl);
            results.push_back(rb);
            std::cout << " acc=" << std::fixed << std::setprecision(4) << rb.accuracy
                      << " F1=" << rb.f1
                      << " (" << (int)rb.train_ms << "ms)\n";
        }
    }

    // ─── Tabla resumen ────────────────────────────────────────
    std::cout << "\n\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        TABLA DE RESULTADOS                            ║\n";
    std::cout << "╠════════════════════════╦═══════════════╦════════╦════════╦═══════════╣\n";
    std::cout << "║ Dataset                ║ Classifier    ║  Acc   ║   F1   ║  Time(ms) ║\n";
    std::cout << "╠════════════════════════╬═══════════════╬════════╬════════╬═══════════╣\n";
    for (auto& r : results) {
        std::cout << "║ " << std::left << std::setw(22) << r.dataset.substr(0,22)
                  << " ║ " << std::setw(13) << r.classifier.substr(0,13)
                  << " ║ " << std::fixed << std::setprecision(4) << r.accuracy
                  << " ║ " << std::setprecision(4) << r.f1
                  << " ║ " << std::setw(9) << (int)r.train_ms << " ║\n";
    }
    std::cout << "╚════════════════════════╩═══════════════╩════════╩════════╩═══════════╝\n";

    // ─── Exportar CSV para análisis ───────────────────────────
    std::ofstream csv("benchmark_results.csv");
    csv << "dataset,classifier,accuracy,f1,train_ms\n";
    for (auto& r : results)
        csv << r.dataset << "," << r.classifier << ","
            << r.accuracy << "," << r.f1 << "," << r.train_ms << "\n";
    std::cout << "\nResultados exportados a benchmark_results.csv\n";
    return 0;
}
