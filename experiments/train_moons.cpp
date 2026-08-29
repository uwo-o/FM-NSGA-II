#include <iostream>
#include <fstream>
#include "nsga2.hpp"
#include "dataset.hpp"

using namespace nsga2;

int main() {
    Dataset data;
    try {
        data.load_csv("../data/moons_2d.csv");
        data.normalize();
    } catch (...) {
        std::cerr << "Error loading moons_2d.csv\n";
        return 1;
    }

    NSGAConfig ncfg;
    ncfg.pop_size = 200; 
    ncfg.max_gen = 600;  
    ncfg.patience = 600; // never stop early
    ncfg.batch_ratio = 1.0; 
    ncfg.p_cross = 0.9;
    ncfg.min_harmonics = 1;
    ncfg.max_harmonics = 12; 
    ncfg.verbose = true;
    
    // ZERO penalty for curve length to wrap the moons perfectly.
    ncfg.obj_cfg.lambda = 0.0001; 
    ncfg.obj_cfg.max_harmonics = 12;
    ncfg.op_cfg.p_coef = 0.2;
    ncfg.op_cfg.p_add = 0.5; // force adding harmonics
    ncfg.op_cfg.p_del = 0.05;

    std::cout << "Entrenando NSGA-Fourier en Two Moons...\n";
    Classifier<FourierManifold> clf(ncfg);
    clf.fit(data);

    if (!clf.fronts().empty() && !clf.fronts()[0].empty()) {
        auto best = clf.fronts()[0][0].manifold;
        std::ofstream f("manifold_moons.csv");
        f << "n_harmonics," << best.n_harmonics << "\n";
        f << "dim," << best.dim << "\n";
        f << "center,";
        for(double c : best.center) f << c << ",";
        f << "\ncoefs,";
        for(double c : best.coefs) f << c << ",";
        f << "\n";
        std::cout << "Guardado manifold_moons.csv\n";
    }

    return 0;
}
