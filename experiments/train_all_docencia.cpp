#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include "dataset.hpp"
#include "metrics.hpp"
#include "mgda_kernels.hpp"
#include "angular_manifold.hpp"
#include "gam_manifold.hpp"
#include "manifold.hpp"
#include "nsga2.hpp"

using namespace nsga2;

// --- Configs from mgda_benchmark ---
struct MGDACfg {
    double lr        = 3e-3;
    double lr_decay  = 0.999;
    int    n_iter    = 2000;
    int    batch_sz  = 128;
    double tol       = 1e-7;
    double lambda_l1 = 0.1;
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

// --- Evaluators ---
template<typename M>
std::pair<double,double> eval_objs(const M& m, const Dataset& data, double lambda_l1) {
    int nw=0;
    for (auto& s:data.samples) if (m.is_inside(s.x)!=(s.label==1)) ++nw;
    double err = (double)nw/data.size();
    double ce=0.; for (auto c:m.coefs) ce+=std::abs(c);
    ce /= m.coefs.size();
    double wl1=0.; for (auto wi:m.w) wl1+=wi;
    wl1 /= m.dim;
    return {err, (1.-lambda_l1)*ce + lambda_l1*wl1};
}
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

template<typename GradFn, typename ApplyFn, typename EvalFn>
void mgda_loop_simple(const MGDACfg& cfg, int param_size, GradFn grad_fn, ApplyFn apply_fn, EvalFn eval_fn, int n_data) {
    mgda::AdamState adam(param_size, cfg.adam_b1, cfg.adam_b2);
    uint32_t rs = 12345u;
    auto rbatch = [&]() -> std::vector<int> {
        if (cfg.batch_sz <= 0) return {};
        std::vector<int> idx(cfg.batch_sz);
        for (auto& i : idx) { rs = rs*1664525u+1013904223u; i=(int)(rs%(uint32_t)n_data); }
        return idx;
    };
    for (int it = 0; it < cfg.n_iter; ++it) {
        double lr = mgda::cosine_lr(it, cfg.lr, cfg.lr_min, cfg.restart_T);
        if (it > 0 && it % cfg.restart_T == 0) adam.reset();
        auto bt = rbatch();
        auto [g1, g2] = grad_fn(bt);
        double alpha = mgda::mgda_alpha(g1, g2);
        auto   d     = mgda::mgda_combine(alpha, g1, g2);
        double gn    = mgda::vec_norm(d);
        if (gn < cfg.tol && it > 50) break;
        if (cfg.use_adam) apply_fn(adam.step(d, lr), 1.0);
        else apply_fn(d, lr);
    }
}

// --- Trainers ---
AngularManifold train_mgda_angular(const Dataset& binary, const std::vector<double>& cen, const MGDACfg& cfg) {
    int dim=(int)binary.samples[0].x.size(), n=binary.size();
    AngularManifold m(dim, cfg.n_harm);
    m.randomize(0.3); m.set_center_near(cen, 0.05);
    int P = 3*dim + cfg.n_harm + 1;
    mgda_loop_simple(cfg, P,
        [&](const std::vector<int>& bt) { return mgda::angular_gradients(m, binary, bt, cfg.lambda_l1, cfg.focal_gamma); },
        [&](const std::vector<double>& d, double lr) { mgda::angular_apply(m, d, lr); },
        [&]() { return eval_objs(m, binary, cfg.lambda_l1); }, n);
    return m;
}

GAMManifold train_mgda_gam(const Dataset& binary, const std::vector<double>& cen, const MGDACfg& cfg) {
    int dim=(int)binary.samples[0].x.size(), n=binary.size();
    GAMManifold m(dim, cfg.n_harm);
    m.randomize(0.5); m.set_center_near(cen, 0.05);
    int P = 1 + dim + dim*2*cfg.n_harm;
    mgda_loop_simple(cfg, P,
        [&](const std::vector<int>& bt) { return mgda::gam_gradients(m, binary, bt, cfg.lambda_l1, cfg.focal_gamma); },
        [&](const std::vector<double>& d, double lr) { mgda::gam_apply(m, d, lr); },
        [&]() { return eval_objs_gam(m, binary, cfg.lambda_l1); }, n);
    return m;
}

// --- Exporters ---
template <typename M>
void export_json(const std::vector<M>& manifolds, const std::vector<int>& labels, const std::string& filename) {
    std::ofstream mj(filename);
    mj << "{\"manifolds\":[\n";
    for (size_t k = 0; k < manifolds.size(); ++k) {
        auto& m = manifolds[k];
        mj << "  {\"class\":" << labels[k] << ",\n"
           << "   \"dim\":" << m.dim << ",\n"
           << "   \"n_harmonics\":" << m.n_harmonics << ",\n";
        
        if constexpr (std::is_same_v<M, AngularManifold>) {
            mj << "   \"type\":\"angular\",\n"
               << "   \"center\":["; for(int i=0;i<m.dim;++i) mj<<m.center[i]<<(i==m.dim-1?"":","); mj<<"],\n"
               << "   \"v\":["; for(int i=0;i<m.dim;++i) mj<<m.v[i]<<(i==m.dim-1?"":","); mj<<"],\n";
        } else if constexpr (std::is_same_v<M, FourierManifold>) {
            mj << "   \"type\":\"fourier\",\n"
               << "   \"center\":["; for(int i=0;i<m.dim;++i) mj<<m.center[i]<<(i==m.dim-1?"":","); mj<<"],\n";
        } else if constexpr (std::is_same_v<M, GAMManifold>) {
            mj << "   \"type\":\"gam\",\n"
               << "   \"beta0\":" << m.beta0 << ",\n";
        }
        
        mj << "   \"w\":["; for(int i=0;i<m.dim;++i) mj << m.w[i] << (i==m.dim-1?"":","); mj << "],\n"
           << "   \"coefs\":["; for(int i=0;i<(int)m.coefs.size();++i) mj << m.coefs[i] << (i==(int)m.coefs.size()-1?"":","); mj << "]}\n";
        if(k < manifolds.size()-1) mj << ",\n";
    }
    mj << "]}\n";
}

int main(int argc, char* argv[]) {
    std::string csv_path = "/home/uwo/Projects/docencia/train_nsga.csv";
    Dataset data;
    data.load_csv(csv_path);
    std::vector<int> labels = {0, 1}; // Docencia labels
    
    // --- MGDA Config ---
    MGDACfg mcfg;
    
    // --- NSGA Config ---
    NSGAConfig ncfg;
    ncfg.pop_size=60; ncfg.max_gen=100; ncfg.patience=30;
    ncfg.batch_ratio=0.5; ncfg.p_cross=0.9;
    ncfg.min_harmonics=1; ncfg.max_harmonics=3;
    ncfg.verbose=true;
    
    std::cout << "Training AM-MGDA...\n";
    std::vector<AngularManifold> am_mgda;
    for (int cls : labels) {
        Dataset binary; std::vector<double> cen(5, 0.); int pc=0;
        for (auto& s : data.samples) {
            Sample bs; bs.x=s.x; bs.label=(s.label==cls)?1:-1; binary.samples.push_back(bs);
            if(s.label==cls){ for(int d=0;d<5;++d) cen[d]+=s.x[d]; ++pc; }
        }
        for(auto& c:cen) c/=pc;
        am_mgda.push_back(train_mgda_angular(binary, cen, mcfg));
    }
    export_json(am_mgda, labels, "manifolds_am_mgda.json");

    std::cout << "Training GAM-MGDA...\n";
    std::vector<GAMManifold> gam_mgda;
    MGDACfg mcfg_gam = mcfg;
    mcfg_gam.lambda_l1 = 0.001; // Avoid w collapse in GAM
    for (int cls : labels) {
        Dataset binary; std::vector<double> cen(5, 0.); int pc=0;
        for (auto& s : data.samples) {
            Sample bs; bs.x=s.x; bs.label=(s.label==cls)?1:-1; binary.samples.push_back(bs);
            if(s.label==cls){ for(int d=0;d<5;++d) cen[d]+=s.x[d]; ++pc; }
        }
        for(auto& c:cen) c/=pc;
        gam_mgda.push_back(train_mgda_gam(binary, cen, mcfg_gam));
    }
    export_json(gam_mgda, labels, "manifolds_gam_mgda.json");

    std::cout << "Training AM-NSGA-II...\n";
    Classifier<AngularManifold> clf_am_nsga(ncfg); clf_am_nsga.fit(data);
    export_json(clf_am_nsga.manifolds(), clf_am_nsga.labels(), "manifolds_am_nsga2.json");

    std::cout << "Training GAM-NSGA-II...\n";
    Classifier<GAMManifold> clf_gam_nsga(ncfg); clf_gam_nsga.fit(data);
    export_json(clf_gam_nsga.manifolds(), clf_gam_nsga.labels(), "manifolds_gam_nsga2.json");

    std::cout << "Training FM-NSGA-II...\n";
    Classifier<FourierManifold> clf_fm_nsga(ncfg); clf_fm_nsga.fit(data);
    export_json(clf_fm_nsga.manifolds(), clf_fm_nsga.labels(), "manifolds_fm_nsga2.json");

    std::cout << "ALL 5 TRAINED AND EXPORTED SUCCESSFULLY!\n";
    return 0;
}
