#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "dataset.hpp"
#include "gam_manifold.hpp"
#include "nsga2.hpp"
#include "mgda_kernels.hpp"

using namespace nsga2;

// ── Configuración MGDA ──
struct MGDACfg {
    double lr        = 1e-2;
    double lr_decay  = 0.999;
    int    n_iter    = 2000;
    int    batch_sz  = 128;
    double tol       = 1e-7;
    double lambda_l1 = 0.5;
    int    n_harm     = 3;
    bool   verbose    = false;
    int    log_every  = 100;
    double fd_eps     = 1e-5;
    double focal_gamma = 2.0;
    bool   use_adam    = true;
    double adam_b1     = 0.9;
    double adam_b2     = 0.999;
    int    restart_T   = 600;
    double lr_min      = 1e-5;
};

// Simple struct to represent feature weights
struct GAMWeights {
    std::string method;
    std::vector<double> w;
};

// ── Objetivos NSGA-II ──
std::pair<double,double> eval_objs_gam(const GAMManifold& m, const Dataset& data, double lambda_l1) {
    int nw=0;
    for (auto& s:data.samples) if (m.is_inside(s.x)!=(s.label==1)) ++nw;
    double err = (double)nw/data.size();
    double ce=0.; for (auto c:m.coefs) ce+=std::abs(c);
    double total = m.coefs.size()+1;
    ce = (ce + std::abs(m.beta0)) / total;
    double wl1=0.; for (auto wi:m.w) wl1+=wi;
    wl1 /= m.dim;
    return {err, (1.-lambda_l1)*ce + lambda_l1*wl1};
}

// Helper: Run MGDA for GAM
GAMWeights explain_mgda(const Dataset& data) {
    int dim = data.n_features;
    GAMManifold m(dim, 3);
    m.randomize(0.5);
    
    MGDACfg cfg;
    
    // Config improved with Adam + Focal
    int P = 1 + dim + dim*2*cfg.n_harm;
    mgda::AdamState adam(P, cfg.adam_b1, cfg.adam_b2);
    
    uint32_t rs = 12345u;
    auto rbatch = [&]() -> std::vector<int> {
        std::vector<int> idx(cfg.batch_sz);
        for(auto& i : idx) { rs = rs*1664525u+1013904223u; i = rs % data.size(); }
        return idx;
    };
    
    for (int it = 0; it < cfg.n_iter; ++it) {
        double lr = mgda::cosine_lr(it, cfg.lr, cfg.lr_min, cfg.restart_T);
        if (it > 0 && it % cfg.restart_T == 0) adam.reset();
        
        auto bt = rbatch();
        auto [g1, g2] = mgda::gam_gradients(m, data, bt, cfg.lambda_l1, cfg.focal_gamma);
        double alpha = mgda::mgda_alpha(g1, g2);
        auto d = mgda::mgda_combine(alpha, g1, g2);
        
        if (mgda::vec_norm(d) < cfg.tol && it > 100) break;
        
        auto d_adam = adam.step(d, lr);
        mgda::gam_apply(m, d_adam, 1.0);
    }
    
    return {"MGDA", m.w};
}

// Helper: Run NSGA-II for GAM
GAMWeights explain_nsga2(const Dataset& data) {
    NSGAConfig cfg;
    cfg.pop_size = 100; cfg.max_gen = 250;
    
    auto obj_fn = [&](const GAMManifold& m, const std::vector<int>& batch) {
        auto objs = eval_objs_gam(m, data, 0.5); // lambda_l1 = 0.5
        return std::array<double, 2>{objs.first, objs.second};
    };
    
    NSGA2<GAMManifold> nsga2(cfg, obj_fn, data.n_features, data.size(), std::vector<double>(data.n_features, 0.0));
    auto pop = nsga2.run();
    
    // Get the best solution (knee point or highest accuracy)
    GAMManifold best_m = pop[0].manifold;
    double best_err = 1.0;
    for (const auto& ind : pop) {
        if (ind.obj[0] < best_err) {
            best_err = ind.obj[0];
            best_m = ind.manifold;
        }
    }
    
    return {"NSGA-II", best_m.w};
}

int main(int argc, char** argv) {
    std::string dataset_path = "../data/breast_cancer_30d.csv";
    if (argc > 1) dataset_path = argv[1];
    
    std::cout << "Loading dataset: " << dataset_path << std::endl;
    Dataset ds;
    ds.load_csv(dataset_path);
    if (ds.size() == 0) {
        std::cerr << "Failed to load dataset." << std::endl;
        return 1;
    }
    
    // We only care about feature importance, so we will transform it into a binary problem
    // class 1 vs rest (Breast Cancer only has 2 classes anyway)
    Dataset binary_ds;
    for(auto s : ds.samples) {
        Sample bs = s;
        bs.label = (s.label == 1) ? 1 : -1;
        binary_ds.samples.push_back(bs);
    }
    binary_ds.n_features = ds.n_features;
    binary_ds.class_labels = {-1, 1};

    std::cout << "Training MGDA-GAM..." << std::endl;
    GAMWeights mw = explain_mgda(binary_ds);
    
    std::cout << "Training NSGA-II-GAM..." << std::endl;
    GAMWeights nw = explain_nsga2(binary_ds);
    
    // Write CSV
    std::ofstream out("gam_weights.csv");
    out << "Feature,Method,Weight\n";
    for(int i = 0; i < ds.n_features; ++i) {
        out << "F" << i << "," << mw.method << "," << mw.w[i] << "\n";
        out << "F" << i << "," << nw.method << "," << nw.w[i] << "\n";
    }
    out.close();
    
    std::cout << "Saved to gam_weights.csv" << std::endl;
    return 0;
}
