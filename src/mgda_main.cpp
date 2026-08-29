// ============================================================
// src/mgda_main.cpp — Demo del clasificador AM-MGDA
// ============================================================
// Compara el clasificador MGDA (gradiente analítico multi-objetivo)
// con los baselines clásicos y el AM-NSGA-II.
// Exporta:
//   mgda_trajectory.csv  — trayectoria completa (iter, error, cpx, alpha, ||d||)
//   mgda_pareto.csv      — punto final de Pareto por clase
// ============================================================
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include "dataset.hpp"
#include "mgda_classifier.hpp"
#include "metrics.hpp"
#include "knn.hpp"
#include "naive_bayes.hpp"
#include "svm.hpp"
#include "decision_tree.hpp"

using namespace nsga2;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    std::string csv_path = "data/moons_2d.csv";
    if (argc > 1) csv_path = argv[1];

    std::cout << "════════════════════════════════════════════════════\n";
    std::cout << "  AM-MGDA Classifier  (Analytic Multi-Gradient)\n";
    std::cout << "════════════════════════════════════════════════════\n\n";

    // ─── Cargar y normalizar dataset ─────────────────────────
    Dataset data;
    try {
        data.load_csv(csv_path);
        data.name = csv_path;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what()
                  << "\nUso: ./mgda <path/to/dataset.csv>\n";
        return 1;
    }
    data.normalize();
    data.print_info();

    auto [train, test] = data.train_test_split(0.2, 42);
    std::cout << "\nTrain: " << train.size()
              << "  Test: "  << test.size()  << "\n\n";

    // ─── Configurar MGDA ─────────────────────────────────────
    mgda::MGDAConfig cfg;
    cfg.lr          = 5e-3;
    cfg.lr_decay    = 0.999;
    cfg.n_iter      = 3000;
    cfg.batch_size  = 128;
    cfg.tol         = 1e-6;
    cfg.lambda_l1   = 0.1;
    cfg.n_harmonics = 3;
    cfg.verbose     = true;
    cfg.log_every   = 200;

    // ─── Entrenar AM-MGDA ────────────────────────────────────
    std::cout << "────────────────────────────────────────────────────\n";
    std::cout << "Training AM-MGDA...\n";
    auto t0 = Clock::now();
    mgda::MGDAClassifier clf(cfg);
    clf.fit(train);
    double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    auto preds = clf.predict_all(test);
    std::vector<int> truth;
    for (auto& s : test.samples) truth.push_back(s.label);

    double acc = multiclass_accuracy(truth, preds);
    double f1  = macro_f1(truth, preds, test.class_labels);

    std::cout << "\n────────────────────────────────────────────────────\n";
    std::cout << "RESULTADOS\n";
    std::cout << "────────────────────────────────────────────────────\n";
    print_report("AM-MGDA", acc, f1, ms);

    // ─── Baselines ───────────────────────────────────────────
    auto run_baseline = [&](IClassifier& bl) {
        auto t  = Clock::now();
        bl.fit(train);
        double bms = std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        auto   p   = bl.predict_all(test);
        double a   = multiclass_accuracy(truth, p);
        double f   = macro_f1(truth, p, test.class_labels);
        print_report(bl.name(), a, f, bms);
    };
    KNN knn3(3), knn7(7);
    GaussianNB nb;
    SVM        svm(1.0, 1.0);
    DecisionTree dt(8);
    run_baseline(knn3); run_baseline(knn7);
    run_baseline(nb);   run_baseline(svm);   run_baseline(dt);
    std::cout << "────────────────────────────────────────────────────\n";

    // ─── Exportar trayectoria MGDA ────────────────────────────
    std::cout << "\n✓ Exportando trayectoria → mgda_trajectory.csv\n";
    std::ofstream traj("mgda_trajectory.csv");
    traj << "clase,iter,error,complexity,alpha,grad_norm\n";
    for (int k = 0; k < (int)clf.labels().size(); ++k) {
        int cls = clf.labels()[k];
        for (auto& tp : clf.trajectories()[k])
            traj << cls    << ","
                 << tp.iter       << ","
                 << tp.error      << ","
                 << tp.complexity << ","
                 << tp.alpha      << ","
                 << tp.grad_norm  << "\n";
    }

    // ─── Exportar puntos finales de Pareto ───────────────────
    std::cout << "✓ Exportando frente Pareto → mgda_pareto.csv\n";
    std::ofstream pf("mgda_pareto.csv");
    pf << "clase,error,complexity,harmonics\n";
    for (int k = 0; k < (int)clf.labels().size(); ++k) {
        int cls = clf.labels()[k];
        auto& t = clf.trajectories()[k];
        if (!t.empty()) {
            auto& last = t.back();
            pf << cls << "," << last.error << "," << last.complexity
               << "," << clf.manifolds()[k].n_harmonics << "\n";
        }
    }

    std::cout << "\n✓ Visualizar: python3 ../scripts/plot_mgda.py\n";
    return 0;
}
