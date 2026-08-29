#include <iostream>
#include <fstream>
#include "nsga2.hpp"
#include "dataset.hpp"
#include "sh_manifold.hpp"

using namespace nsga2;

template<typename M>
void export_pareto_front(const std::vector<Individual<M>>& front, const std::string& filename) {
    std::ofstream out(filename);
    out << "rank,error,complejidad\n";
    for (const auto& ind : front) {
        out << ind.rank << "," << ind.obj[0] << "," << ind.obj[1] << "\n";
    }
}

int main() {
    Dataset data;
    try {
        data.load_csv("../data/moons_2d.csv");
        data.normalize();
    } catch (...) {
        std::cerr << "Error loading moons_2d.csv\n";
        return 1;
    }
    auto [train, test] = data.train_test_split(0.2, 42);

    NSGAConfig cfg;
    cfg.pop_size = 60;
    cfg.max_gen = 150;
    cfg.patience = 40;
    cfg.batch_ratio = 0.3;
    cfg.p_cross = 0.9;
    cfg.min_harmonics = 1;
    cfg.max_harmonics = 5;
    cfg.verbose = true;
    cfg.obj_cfg.max_harmonics = 5;

    std::cout << "Entrenando SHM en Two Moons para extraer Frente Pareto...\n";
    Classifier<SphericalHarmonicManifold> clf(cfg);
    clf.fit(train);

    if (!clf.fronts().empty()) {
        export_pareto_front(clf.fronts()[0], "pareto_front_NSGA-SHM.csv");
        std::cout << "Guardado pareto_front_NSGA-SHM.csv\n";
    }

    return 0;
}
