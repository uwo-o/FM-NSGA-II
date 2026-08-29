// ============================================================
// experiments/mgda_benchmark.cpp — Benchmark MGDA 3 kernels
// ============================================================
//
// Compara los 3 kernels topológicos entrenados con MGDA
// frente a los mismos kernels entrenados con NSGA-II y los
// baselines deterministas, en todos los datasets disponibles.
//
// Exporta mgda_bench_results.csv para análisis posterior.
// ============================================================
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cassert>
#include "dataset.hpp"
#include "metrics.hpp"
#include "mgda_kernels.hpp"
#include "angular_manifold.hpp"
#include "gam_manifold.hpp"
#include "manifold.hpp"
#include "knn.hpp"
#include "naive_bayes.hpp"
#include "svm.hpp"
#include "decision_tree.hpp"

// NSGA-II para comparar
#include "nsga2.hpp"

using namespace nsga2;
using Clock = std::chrono::high_resolution_clock;

// ─── Estructura de resultado ──────────────────────────────────
struct Result {
    std::string dataset, classifier;
    double accuracy, f1, train_ms;
};

// ─── Punto de trayectoria ─────────────────────────────────────
struct TrajPt {
    int iter;
    double error, complexity, alpha, grad_norm;
};

// ─── Configuración MGDA ───────────────────────────────────────
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
    double fd_eps     = 1e-5;    // finitas para Fourier
    // ── Mejoras anti-mínimos-locales ──
    double focal_gamma = 2.0;   // Focal Loss: 0=logistic, 2=focal estándar
    bool   use_adam    = true;   // Adam optimizer
    double adam_b1     = 0.9;
    double adam_b2     = 0.999;
    int    restart_T   = 600;   // Cosine warm restart cada T iteraciones
    double lr_min      = 1e-5;  // lr mínimo del coseno
};

// ─── Evaluación de objetivos (tasa de error 0-1 y complejidad) ─
template<typename M>
std::pair<double,double> eval_objs(const M& m, const Dataset& data, double lambda_l1) {
    int nw=0;
    for (auto& s:data.samples) if (m.is_inside(s.x)!=(s.label==1)) ++nw;
    double err = (double)nw/data.size();
    // complejidad: norma de coefs + L1 de w
    double ce=0.; for (auto c:m.coefs) ce+=std::abs(c);
    ce /= m.coefs.size();
    double wl1=0.; for (auto wi:m.w) wl1+=wi;
    wl1 /= m.dim;
    return {err, (1.-lambda_l1)*ce + lambda_l1*wl1};
}
// Especialización para GAMManifold (tiene beta0 aparte)
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

// ─── MACRO helper: loop MGDA con Adam + Focal + Cosine Restarts ─────────────────────
// Separa la lógica de optimización del cálculo de gradientes.
// Argumentos:
//   m         — manifold a optimizar (modificado in-place)
//   grad_fn   — lambda(batch) -> pair<g1,g2>
//   apply_fn  — lambda(d, lr) -> void
//   eval_fn   — lambda() -> pair<error, complejidad>
// ─────────────────────────────────────────────────────────────
template<typename GradFn, typename ApplyFn, typename EvalFn>
std::vector<TrajPt> mgda_loop(
    const MGDACfg& cfg, int param_size,
    GradFn grad_fn, ApplyFn apply_fn, EvalFn eval_fn,
    int n_data, uint32_t seed = 12345u)
{
    std::vector<TrajPt> traj;
    int bsz = (cfg.batch_sz > 0 && cfg.batch_sz < n_data) ? cfg.batch_sz : 0;

    // Adam state para la dirección combinada d
    mgda::AdamState adam(param_size, cfg.adam_b1, cfg.adam_b2);

    uint32_t rs = seed;
    auto rbatch = [&]() -> std::vector<int> {
        if (bsz <= 0) return {};
        std::vector<int> idx(bsz);
        for (auto& i : idx) { rs = rs*1664525u+1013904223u; i=(int)(rs%(uint32_t)n_data); }
        return idx;
    };

    for (int it = 0; it < cfg.n_iter; ++it) {
        // Cosine LR con warm restarts
        double lr = mgda::cosine_lr(it, cfg.lr, cfg.lr_min, cfg.restart_T);
        // Reset Adam en cada restart para que "arranque" del nuevo lr alto
        if (it > 0 && it % cfg.restart_T == 0) adam.reset();

        auto bt = rbatch();
        auto [g1, g2] = grad_fn(bt);

        double alpha = mgda::mgda_alpha(g1, g2);
        auto   d     = mgda::mgda_combine(alpha, g1, g2);
        double gn    = mgda::vec_norm(d);

        // Log periódico
        if (it % cfg.log_every == 0 || it == cfg.n_iter-1) {
            auto [e, c] = eval_fn();
            traj.push_back({it, e, c, alpha, gn});
        }
        // Detección de punto Pareto
        if (gn < cfg.tol && it > 50) {
            auto [e, c] = eval_fn();
            traj.push_back({it, e, c, alpha, gn});
            break;
        }

        // Actualización: Adam o SGD puro
        if (cfg.use_adam) {
            auto d_adam = adam.step(d, lr);
            apply_fn(d_adam, 1.0);   // lr ya aplicado por Adam
        } else {
            apply_fn(d, lr);
        }
    }
    return traj;
}

// ─── Angular ───────────────────────────────────────────────────────────
std::pair<AngularManifold, std::vector<TrajPt>>
train_mgda_angular(const Dataset& binary, const std::vector<double>& cen,
                   const MGDACfg& cfg)
{
    int dim=(int)binary.samples[0].x.size(), n=binary.size();
    AngularManifold m(dim, cfg.n_harm);
    m.randomize(0.3); m.set_center_near(cen, 0.05);
    int P = 3*dim + cfg.n_harm + 1;

    auto traj = mgda_loop(cfg, P,
        [&](const std::vector<int>& bt) {
            return mgda::angular_gradients(m, binary, bt, cfg.lambda_l1, cfg.focal_gamma);
        },
        [&](const std::vector<double>& d, double lr) {
            mgda::angular_apply(m, d, lr);
        },
        [&]() { return eval_objs(m, binary, cfg.lambda_l1); },
        n, 12345u);

    return {m, traj};
}


// ─── GAM ────────────────────────────────────────────────────────────
std::pair<GAMManifold, std::vector<TrajPt>>
train_mgda_gam(const Dataset& binary, const std::vector<double>& cen,
               const MGDACfg& cfg)
{
    int dim=(int)binary.samples[0].x.size(), n=binary.size();
    GAMManifold m(dim, cfg.n_harm);
    m.randomize(0.5); m.set_center_near(cen, 0.05);
    int P = 1 + dim + dim*2*cfg.n_harm;

    auto traj = mgda_loop(cfg, P,
        [&](const std::vector<int>& bt) {
            return mgda::gam_gradients(m, binary, bt, cfg.lambda_l1, cfg.focal_gamma);
        },
        [&](const std::vector<double>& d, double lr) {
            mgda::gam_apply(m, d, lr);
        },
        [&]() { return eval_objs_gam(m, binary, cfg.lambda_l1); },
        n, 54321u);

    return {m, traj};
}

// ─── Fourier (diferencias finitas) ─────────────────────────────────────
std::pair<FourierManifold, std::vector<TrajPt>>
train_mgda_fourier(const Dataset& binary, const std::vector<double>& cen,
                   const MGDACfg& cfg)
{
    int dim=(int)binary.samples[0].x.size(), n=binary.size();
    FourierManifold m(dim, cfg.n_harm);
    m.randomize(0.3); m.set_center_near(cen, 0.05);
    int Pc = 2*cfg.n_harm*dim;
    int P  = 2*dim + Pc;
    int bsz = std::min(cfg.batch_sz/2, 32);

    // Cfg reducida para Fourier: batch pequeño, fd_eps del config
    MGDACfg fcfg = cfg;
    fcfg.batch_sz = bsz;

    auto traj = mgda_loop(fcfg, P,
        [&](const std::vector<int>& bt) {
            return mgda::fourier_gradients(m, binary, bt, cfg.lambda_l1, cfg.fd_eps, cfg.focal_gamma);
        },
        [&](const std::vector<double>& d, double lr) {
            mgda::fourier_apply(m, d, lr);
        },
        [&]() { return eval_objs(m, binary, cfg.lambda_l1); },
        n, 99999u);

    return {m, traj};
}

// ─── Wrapper multiclase One-vs-Rest ──────────────────────────
template<typename M, typename TrainFn>
Result run_mgda_ovr(const std::string& ds_name, const std::string& clf_name,
                    const Dataset& train, const Dataset& test,
                    TrainFn train_fn)
{
    auto t0 = Clock::now();
    std::vector<M>   manifolds;
    std::vector<int> labels = train.class_labels;

    for (int cls : labels) {
        Dataset binary;
        std::vector<double> cen(train.samples[0].x.size(), 0.);
        int pc=0;
        for (auto& s : train.samples) {
            Sample bs; bs.x=s.x; bs.label=(s.label==cls)?1:-1;
            binary.samples.push_back(bs);
            if(s.label==cls){ for(int d=0;d<(int)s.x.size();++d) cen[d]+=s.x[d]; ++pc; }
        }
        binary.class_labels={-1,1};
        if(pc>0) for(auto& c:cen) c/=pc;
        auto [mf, _] = train_fn(binary, cen);
        manifolds.push_back(mf);
        
        // --- EXTRA: EXPORT TWO MOONS MANIFOLD FOR VISUALIZATION ---
        // Export only for class 1 to avoid overwriting and match NSGA-II export style
        if constexpr (std::is_same_v<M, nsga2::FourierManifold>) {
            if (cls == 1 && ds_name == "Two Moons (2D)" && clf_name == "MGDA-Fourier") {
                std::ofstream f("manifold_moons_mgda.csv");
                f << "n_harmonics," << mf.n_harmonics << "\n";
                f << "dim," << mf.dim << "\n";
                f << "center,";
                for(double c : mf.center) f << c << ",";
                f << "\ncoefs,";
                for(double c : mf.coefs) f << c << ",";
                f << "\n";
            }
        }
    }
    double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();

    // Predicción: clase con menor distancia con signo
    std::vector<int> preds, truth;
    for (auto& s : test.samples) {
        double best=1e18; int best_c=labels[0];
        for(int k=0;k<(int)labels.size();++k){
            double d=manifolds[k].signed_distance(s.x);
            if(d<best){best=d;best_c=labels[k];}
        }
        preds.push_back(best_c);
        truth.push_back(s.label);
    }
    return {ds_name, clf_name,
            multiclass_accuracy(truth,preds),
            macro_f1(truth,preds,test.class_labels),
            ms};
}

// ─── Baseline genérico ────────────────────────────────────────
Result run_baseline(const std::string& ds, const Dataset& tr,
                    const Dataset& te, IClassifier& bl)
{
    auto t0=Clock::now();
    bl.fit(tr);
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    auto p=bl.predict_all(te);
    std::vector<int> truth; for(auto& s:te.samples) truth.push_back(s.label);
    return {ds, bl.name(), multiclass_accuracy(truth,p), macro_f1(truth,p,te.class_labels), ms};
}

// ─── NSGA-II para comparar (usa Classifier<M>) ────────────────
template<typename M>
void export_pareto_front(const std::vector<Individual<M>>& front, const std::string& filename) {
    std::ofstream out(filename);
    out << "rank,error,complejidad\n";
    for (const auto& ind : front) {
        out << ind.rank << "," << ind.obj[0] << "," << ind.obj[1] << "\n";
    }
}

template<typename M>
Result run_nsga2(const std::string& ds, const std::string& name,
                 const Dataset& tr, const Dataset& te,
                 const NSGAConfig& cfg)
{
    auto t0=Clock::now();
    Classifier<M> clf(cfg); clf.fit(tr);
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    
    // Export pareto for Two Moons (class 0 for simplicity)
    if (ds == "Two Moons (2D)" && !clf.fronts().empty()) {
        export_pareto_front(clf.fronts()[0], "pareto_front_" + name + ".csv");
        if constexpr (std::is_same_v<M, nsga2::FourierManifold>) {
            if (name == "NSGA-Fourier" && !clf.fronts()[0].empty()) {
                auto best = clf.fronts()[0][0].manifold;
                std::ofstream f("manifold_moons.csv");
                f << "n_harmonics," << best.n_harmonics << "\n";
                f << "dim," << best.dim << "\n";
                f << "center,";
                for(double c : best.center) f << c << ",";
                f << "\ncoefs,";
                for(double c : best.coefs) f << c << ",";
                f << "\n";
            }
        }
    }
    
    auto p=clf.predict_all(te);
    std::vector<int> truth; for(auto& s:te.samples) truth.push_back(s.label);
    return {ds, name, multiclass_accuracy(truth,p), macro_f1(truth,p,te.class_labels), ms};
}

// ─────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        BENCHMARK MGDA — 3 Kernels Topológicos                   ║\n";
    std::cout << "║  Fourier (FD-grad)  ·  Angular (analytic)  ·  GAM (analytic)   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // ─── Datasets ─────────────────────────────────────────────
    struct DS { std::string name, path; };
    std::vector<DS> datasets = {
        {"Two Moons (2D)",    "../data/moons_2d.csv"},
        {"Circles (2D)",      "../data/circles_2d.csv"},
        {"Spirals (2D)",      "../data/spirals_2d.csv"},
        {"Blobs (2D)",        "../data/blobs_2d.csv"},
        {"Chessboard (2D)",   "../data/chessboard_2d.csv"},
        {"Adversarial(52D)",  "../data/adversarial_52d.csv"},
        {"Iris (4D)",         "../data/iris.csv"},
        {"Wine (13D)",        "../data/wine.csv"},
        {"BreastCancer(30D)", "../data/breast_cancer.csv"}
    };

    // ─── Configuración MGDA ────────────────────────────────────
    MGDACfg mcfg;
    mcfg.lr=3e-3; mcfg.lr_decay=0.999; mcfg.n_iter=2000;
    mcfg.batch_sz=128; mcfg.tol=1e-7; mcfg.lambda_l1=0.1;
    mcfg.n_harm=3; mcfg.verbose=false; mcfg.log_every=200;
    mcfg.fd_eps=1e-5;

    // Config reducida para Fourier-MGDA (finitas son muy lentas en alta dim)
    MGDACfg mcfg_fourier = mcfg;
    mcfg_fourier.n_iter = 500;    // límite duro: sin esto tarda horas en dim>4
    mcfg_fourier.batch_sz = 32;   // batch mínimo para que las finitas sean tolerables

    // ─── Configuración NSGA-II (rápida, para comparar) ─────────
    NSGAConfig ncfg;
    ncfg.pop_size=60; ncfg.max_gen=150; ncfg.patience=40;
    ncfg.batch_ratio=0.3; ncfg.p_cross=0.9;
    ncfg.min_harmonics=1; ncfg.max_harmonics=8;
    ncfg.verbose=false;
    ncfg.obj_cfg.lambda=0.5; ncfg.obj_cfg.max_harmonics=8;
    ncfg.op_cfg.p_coef=0.12; ncfg.op_cfg.p_add=0.06; ncfg.op_cfg.p_del=0.04;

    std::vector<Result> all;

    for (auto& ds : datasets) {
        Dataset data;
        try { data.load_csv(ds.path); }
        catch (...) { std::cerr << "⚠ Skipping " << ds.name << "\n"; continue; }
        data.normalize();
        auto [tr,te] = data.train_test_split(0.2, 42);

        std::cout << "\n═══ " << ds.name
                  << " (n=" << data.size()
                  << " d=" << data.n_features
                  << " K=" << data.n_classes << ") ═══\n";

        // ── MGDA kernels ──────────────────────────────────────
        std::cout << "  [MGDA-Angular]..." << std::flush;
        auto ra = run_mgda_ovr<AngularManifold>(ds.name, "MGDA-Angular", tr, te,
            [&](const Dataset& bin, const std::vector<double>& cen){
                return train_mgda_angular(bin, cen, mcfg);
            });
        all.push_back(ra);
        std::cout << " acc=" << std::fixed << std::setprecision(4) << ra.accuracy
                  << " (" << (int)ra.train_ms << "ms)\n";

        std::cout << "  [MGDA-GAM]..." << std::flush;
        auto rg = run_mgda_ovr<GAMManifold>(ds.name, "MGDA-GAM", tr, te,
            [&](const Dataset& bin, const std::vector<double>& cen){
                return train_mgda_gam(bin, cen, mcfg);
            });
        all.push_back(rg);
        std::cout << " acc=" << std::fixed << std::setprecision(4) << rg.accuracy
                  << " (" << (int)rg.train_ms << "ms)\n";


        if (ds.name == "Two Moons (2D)") {
            std::cout << "  [MGDA-Fourier]..." << std::flush;
            auto rf = run_mgda_ovr<FourierManifold>(ds.name, "MGDA-Fourier", tr, te,
                [&](const Dataset& bin, const std::vector<double>& cen){
                    return train_mgda_fourier(bin, cen, mcfg_fourier);
                });
            all.push_back(rf);
            std::cout << " acc=" << std::fixed << std::setprecision(4) << rf.accuracy
                      << " (" << (int)rf.train_ms << "ms)\n";
        }
        
        // ── NSGA-II kernels ────────────────────────────────────        // Angular NSGA-II
        std::cout << "  [NSGA-Angular]..." << std::flush;
        auto na = run_nsga2<AngularManifold>(ds.name, "NSGA-Angular", tr, te, ncfg);
        all.push_back(na);
        std::cout << " acc=" << na.accuracy << " (" << (int)na.train_ms << "ms)\n";

        // GAM NSGA-II
        std::cout << "  [NSGA-GAM]..." << std::flush;
        auto ng = run_nsga2<GAMManifold>(ds.name, "NSGA-GAM", tr, te, ncfg);
        all.push_back(ng);
        std::cout << " acc=" << ng.accuracy << " (" << (int)ng.train_ms << "ms)\n";

        // Fourier NSGA-II
        std::cout << "  [NSGA-Fourier]..." << std::flush;
        auto nf = run_nsga2<FourierManifold>(ds.name, "NSGA-Fourier", tr, te, ncfg);
        all.push_back(nf);
        std::cout << " acc=" << nf.accuracy << " (" << (int)nf.train_ms << "ms)\n";

        // ── Baselines ─────────────────────────────────────────
        KNN knn3(3), knn7(7);
        GaussianNB nb;
        SVM svm(1.0,1.0);
        DecisionTree dt(8);
        for (auto* bl : std::vector<IClassifier*>{&knn3,&knn7,&nb,&svm,&dt}) {
            auto rb = run_baseline(ds.name, tr, te, *bl);
            all.push_back(rb);
            std::cout << "  [" << bl->name() << "] acc=" << rb.accuracy << "\n";
        }
    }

    // ─── Tabla resumen ────────────────────────────────────────
    std::cout << "\n\n";
    std::cout << "╔═══════════════════════╦════════════════╦════════╦════════╦══════════╗\n";
    std::cout << "║ Dataset               ║ Classifier     ║  Acc   ║   F1   ║ Time(ms) ║\n";
    std::cout << "╠═══════════════════════╬════════════════╬════════╬════════╬══════════╣\n";
    std::string prev="";
    for (auto& r:all) {
        if(r.dataset!=prev&&!prev.empty())
            std::cout << "╟───────────────────────╫────────────────╫────────╫────────╫──────────╢\n";
        prev=r.dataset;
        std::cout << "║ " << std::left  << std::setw(21) << r.dataset.substr(0,21)
                  << " ║ " << std::setw(14) << r.classifier.substr(0,14)
                  << " ║ " << std::right << std::fixed << std::setprecision(4) << r.accuracy
                  << " ║ " << std::setprecision(4) << r.f1
                  << " ║ " << std::setw(8) << (int)r.train_ms << " ║\n";
    }
    std::cout << "╚═══════════════════════╩════════════════╩════════╩════════╩══════════╝\n";

    // ─── Exportar CSV ─────────────────────────────────────────
    std::ofstream csv("mgda_bench_results.csv");
    csv << "dataset,classifier,accuracy,f1,train_ms\n";
    for (auto& r:all)
        csv << r.dataset << "," << r.classifier << ","
            << r.accuracy << "," << r.f1 << "," << r.train_ms << "\n";
    std::cout << "\n✓ Exportado: mgda_bench_results.csv\n";
    return 0;
}
