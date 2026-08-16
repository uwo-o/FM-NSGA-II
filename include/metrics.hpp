#pragma once
// ============================================================
// include/metrics.hpp — Métricas de evaluación
// ============================================================
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "dataset.hpp"

namespace nsga2 {

struct ClassificationReport {
    double accuracy;
    double precision;
    double recall;
    double f1;
    int    n_samples;
    std::vector<std::vector<int>> confusion; // [true][pred]
};

// ─── Métricas para clasificación binaria (labels 0/1) ────────
inline ClassificationReport evaluate_binary(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred) {
    ClassificationReport r;
    r.n_samples = (int)y_true.size();
    r.confusion  = {{0,0},{0,0}};

    int tp=0,fp=0,fn=0,tn=0;
    for (int i = 0; i < r.n_samples; ++i) {
        int t = y_true[i], p = y_pred[i];
        r.confusion[t][p]++;
        if (t==1 && p==1) ++tp;
        else if (t==0 && p==1) ++fp;
        else if (t==1 && p==0) ++fn;
        else ++tn;
    }
    r.accuracy  = static_cast<double>(tp+tn) / r.n_samples;
    r.precision = (tp+fp)>0 ? static_cast<double>(tp)/(tp+fp) : 0.0;
    r.recall    = (tp+fn)>0 ? static_cast<double>(tp)/(tp+fn) : 0.0;
    r.f1        = (r.precision+r.recall)>0
                  ? 2*r.precision*r.recall/(r.precision+r.recall) : 0.0;
    return r;
}

// ─── Accuracy multi-clase ─────────────────────────────────────
inline double multiclass_accuracy(const std::vector<int>& y_true,
                                   const std::vector<int>& y_pred) {
    if (y_true.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i]) ++correct;
    return static_cast<double>(correct) / y_true.size();
}

// ─── F1 macro-promedio ────────────────────────────────────────
inline double macro_f1(const std::vector<int>& y_true,
                        const std::vector<int>& y_pred,
                        const std::vector<int>& labels) {
    double sum = 0.0;
    for (int lbl : labels) {
        // one-vs-rest
        std::vector<int> t, p;
        for (size_t i = 0; i < y_true.size(); ++i) {
            t.push_back(y_true[i]==lbl ? 1 : 0);
            p.push_back(y_pred[i]==lbl ? 1 : 0);
        }
        auto rep = evaluate_binary(t, p);
        sum += rep.f1;
    }
    return sum / labels.size();
}

// ─── Impresión de resultados ──────────────────────────────────
inline void print_report(const std::string& name, double acc, double f1,
                          double train_ms = -1) {
    std::cout << std::left << std::setw(28) << name
              << "  acc="  << std::fixed << std::setprecision(4) << acc
              << "  F1="   << std::fixed << std::setprecision(4) << f1;
    if (train_ms >= 0)
        std::cout << "  t=" << std::setprecision(1) << train_ms << "ms";
    std::cout << "\n";
}

} // namespace nsga2
