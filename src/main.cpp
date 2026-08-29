// ============================================================
// src/main.cpp — Demo rápido del clasificador FM-NSGA-II
// ============================================================
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include "dataset.hpp"
#include "nsga2.hpp"
#include "angular_manifold.hpp"
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
    std::cout << "  FM-NSGA-II Classifier\n";
    std::cout << "════════════════════════════════════════════════════\n\n";

    // ─── Cargar dataset ──────────────────────────────────────
    Dataset data;
    try {
        data.load_csv(csv_path);
        data.name = csv_path;
    } catch (std::exception& e) {
        std::cerr << "Error cargando dataset: " << e.what() << "\n";
        std::cerr << "Uso: ./fm_nsga2 <path/to/dataset.csv>\n";
        return 1;
    }
    data.normalize();
    data.print_info();

    auto [train, test] = data.train_test_split(0.2, 42);
    std::cout << "\nTrain: " << train.size() << "  Test: " << test.size() << "\n\n";

    // ─── Configuración NSGA-II ───────────────────────────────────
    NSGAConfig cfg;
    cfg.pop_size      = 80;
    cfg.max_gen       = 200;
    cfg.p_cross       = 0.9;
    cfg.min_harmonics = 1;
    cfg.max_harmonics = 8;
    cfg.verbose       = true;
    cfg.log_every     = 50;
    cfg.op_cfg.p_coef    = 0.15;
    cfg.op_cfg.p_add     = 0.08;  // mayor prob de agregar armónicos
    cfg.op_cfg.p_del     = 0.03;  // menor prob de eliminar armónicos
    cfg.obj_cfg.lambda   = 0.2;   // baja penalización de complejidad
    cfg.obj_cfg.arc_baseline = 8.0;

    // ─── Entrenar Classifier<AngularManifold> ─────────────────────────
    std::cout << "────────────────────────────────────────────────────\n";
    std::cout << "Entrenando FM-NSGA-II Classifier...\n";
    auto t0 = Clock::now();
    Classifier<AngularManifold> clf(cfg);
    clf.fit(train);
    double nsga_ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    auto preds = clf.predict_all(test);
    std::vector<int> truth;
    for (auto& s : test.samples) truth.push_back(s.label);

    double acc_nsga = multiclass_accuracy(truth, preds);
    double f1_nsga  = macro_f1(truth, preds, test.class_labels);

    std::cout << "\n────────────────────────────────────────────────────\n";
    std::cout << "RESULTADOS EN TEST SET\n";
    std::cout << "────────────────────────────────────────────────────\n";
    print_report("FM-NSGA-II", acc_nsga, f1_nsga, nsga_ms);

    // ─── Baselines ───────────────────────────────────────────
    auto run_baseline = [&](IClassifier& bl) {
        auto t = Clock::now();
        bl.fit(train);
        double ms = std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        auto   p  = bl.predict_all(test);
        double a  = multiclass_accuracy(truth, p);
        double f  = macro_f1(truth, p, test.class_labels);
        print_report(bl.name(), a, f, ms);
    };

    KNN          knn3(3),  knn7(7);
    GaussianNB   nb;
    SVM          svm(1.0, 1.0);
    DecisionTree dt(8);

    run_baseline(knn3);
    run_baseline(knn7);
    run_baseline(nb);
    run_baseline(svm);
    run_baseline(dt);

    std::cout << "\nManifolds aprendidos:\n";
    std::ofstream mj("angular_manifolds_nsga.json");
    mj << "{\"manifolds\":[\n";
    for (size_t k = 0; k < clf.manifolds().size(); ++k) {
        auto& m = clf.manifolds()[k];
        std::cout << "  Clase " << clf.labels()[k]
                  << ": " << m.n_harmonics << " harmónicos"
                  << "  arco=" << std::fixed << std::setprecision(3) << m.arc_length()
                  << "\n";
        mj << "  {\"class\":" << clf.labels()[k] << ",\n"
           << "   \"dim\":" << m.dim << ",\n"
           << "   \"n_harmonics\":" << m.n_harmonics << ",\n"
           << "   \"center\":["; for(int i=0;i<m.dim;++i) mj << m.center[i] << (i==m.dim-1?"":","); mj << "],\n"
           << "   \"v\":["; for(int i=0;i<m.dim;++i) mj << m.v[i] << (i==m.dim-1?"":","); mj << "],\n"
           << "   \"w\":["; for(int i=0;i<m.dim;++i) mj << m.w[i] << (i==m.dim-1?"":","); mj << "],\n"
           << "   \"coefs\":["; for(int i=0;i<(int)m.coefs.size();++i) mj << m.coefs[i] << (i==(int)m.coefs.size()-1?"":","); mj << "]}\n";
        if(k < clf.manifolds().size()-1) mj << ",\n";
    }
    mj << "]}\n";
    std::cout << "\n✓ Exportando frente de Pareto...\n";
    std::ofstream ofs("pareto_front.csv");
    ofs << "clase,error,complejidad,rank\n";
    for (size_t k = 0; k < clf.fronts().size(); ++k) {
        int cls = clf.labels()[k];
        for (const auto& ind : clf.fronts()[k]) {
            ofs << cls << "," << ind.obj[0] << "," << ind.obj[1] << "," << ind.rank << "\n";
        }
    }
    
    std::cout << "\n✓ Visualizar: python3 visualize.py (solo para datasets 2D)\n";
    return 0;
}
