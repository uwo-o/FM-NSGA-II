#include <iostream>
#include <iomanip>
#include <vector>
#include "dataset.hpp"
#include "nsga2.hpp"

using namespace nsga2;

int main() {
    Dataset moons; moons.load_csv("../data/moons_2d.csv");
    
    NSGAConfig cfg;
    cfg.pop_size = 200;
    cfg.max_gen = 100;
    cfg.min_harmonics = 2;
    cfg.max_harmonics = 2; // H=2
    cfg.verbose = false;
    
    FMClassifier clf(cfg);
    clf.fit(moons);
    
    auto manifolds = clf.manifolds();
    auto m = manifolds[0]; // Class 0
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "center = [" << m.center[0] << ", " << m.center[1] << "]\n";
    
    // coefs layout: [a_{0,k}, b_{0,k}, a_{1,k}, b_{1,k}] for k=1..H
    for (int k = 1; k <= m.n_harmonics; ++k) {
        int idx = (k - 1) * 4;
        double ax = m.coefs[idx];
        double bx = m.coefs[idx + 1];
        double ay = m.coefs[idx + 2];
        double by = m.coefs[idx + 3];
        std::cout << "k=" << k << " -> ax=" << ax << ", bx=" << bx 
                  << ", ay=" << ay << ", by=" << by << "\n";
    }
    return 0;
}
