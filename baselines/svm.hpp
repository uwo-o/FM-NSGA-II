#pragma once
// ============================================================
// baselines/svm.hpp — SVM con kernel RBF (SMO simplificado)
//                     Multi-clase via one-vs-rest
// ============================================================
// Implementación del algoritmo SMO (Platt, 1998):
//   Dual: max Σα_i - ½ Σ_{i,j} α_i α_j y_i y_j K(x_i, x_j)
//   sujeto a: 0 ≤ α_i ≤ C,  Σ α_i y_i = 0
//   Kernel RBF: K(x,z) = exp(-γ ||x-z||²)
//   Predicción: f(x) = Σ α_i y_i K(x_i, x) + b
// ============================================================
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include "classifier_base.hpp"
#include "../include/utils.hpp"

namespace nsga2 {

// ─── SVM Binario (labels: +1 / -1) ───────────────────────────
class BinarySVM {
public:
    BinarySVM(double C = 1.0, double gamma = 1.0, int max_iter = 200, std::string kernel_type = "rbf")
        : C_(C), gamma_(gamma), max_iter_(max_iter), kernel_type_(kernel_type) {}

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y) {
        X_    = X;
        y_    = y;
        int n = (int)X.size();
        alpha_.assign(n, 0.0);
        b_ = 0.0;

        // Cache del kernel
        std::vector<std::vector<double>> K(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j)
                K[i][j] = K[j][i] = kernel(X[i], X[j]);

        // SMO simplificado
        int iter = 0;
        bool changed = true;
        while (changed && iter < max_iter_) {
            changed = false;
            for (int i = 0; i < n; ++i) {
                double Ei = decision_raw(X[i]) - y[i];
                if ((y[i]*Ei < -1e-3 && alpha_[i] < C_) ||
                    (y[i]*Ei >  1e-3 && alpha_[i] > 0.0)) {
                    // Selección heurística de j
                    int j = (i + 1 + rand_int(0, n-1)) % n;
                    if (j == i) continue;
                    double Ej = decision_raw(X[j]) - y[j];

                    double ai_old = alpha_[i], aj_old = alpha_[j];
                    double L, H;
                    if (y[i] == y[j]) {
                        L = std::max(0.0, ai_old + aj_old - C_);
                        H = std::min(C_,  ai_old + aj_old);
                    } else {
                        L = std::max(0.0, aj_old - ai_old);
                        H = std::min(C_,  C_ + aj_old - ai_old);
                    }
                    if (H - L < 1e-10) continue;

                    double eta = 2*K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) continue;

                    alpha_[j] -= y[j]*(Ei - Ej) / eta;
                    alpha_[j]  = std::clamp(alpha_[j], L, H);
                    if (std::abs(alpha_[j] - aj_old) < 1e-5) continue;

                    alpha_[i] += y[i]*y[j]*(aj_old - alpha_[j]);

                    // Actualizar bias
                    double b1 = b_ - Ei - y[i]*(alpha_[i]-ai_old)*K[i][i]
                                       - y[j]*(alpha_[j]-aj_old)*K[i][j];
                    double b2 = b_ - Ej - y[i]*(alpha_[i]-ai_old)*K[i][j]
                                       - y[j]*(alpha_[j]-aj_old)*K[j][j];
                    if (0 < alpha_[i] && alpha_[i] < C_)      b_ = b1;
                    else if (0 < alpha_[j] && alpha_[j] < C_) b_ = b2;
                    else b_ = (b1 + b2) / 2.0;

                    changed = true;
                }
            }
            ++iter;
        }

        // Guardar solo los support vectors
        sv_.clear(); sv_alpha_.clear(); sv_y_.clear();
        for (int i = 0; i < n; ++i)
            if (alpha_[i] > 1e-8) {
                sv_.push_back(X[i]);
                sv_alpha_.push_back(alpha_[i]);
                sv_y_.push_back(y[i]);
            }
    }

    // f(x) > 0 → clase +1, < 0 → clase -1
    double decision_function(const std::vector<double>& x) const {
        double s = b_;
        for (size_t i = 0; i < sv_.size(); ++i)
            s += sv_alpha_[i] * sv_y_[i] * kernel(sv_[i], x);
        return s;
    }

    int classify(const std::vector<double>& x) const {
        return decision_function(x) >= 0 ? 1 : -1;
    }

private:
    double C_, gamma_;
    int    max_iter_;
    std::string kernel_type_;
    std::vector<std::vector<double>> X_, sv_;
    std::vector<double> alpha_, sv_alpha_;
    std::vector<int>    y_, sv_y_;
    double b_ = 0.0;

    double kernel(const std::vector<double>& a, const std::vector<double>& b) const {
        if (kernel_type_ == "linear") {
            double dot = 0.0;
            for(size_t i=0; i<a.size(); ++i) dot += a[i]*b[i];
            return dot;
        }
        return std::exp(-gamma_ * dist2_sq(a, b));
    }

    double decision_raw(const std::vector<double>& x) const {
        double s = b_;
        for (size_t i = 0; i < X_.size(); ++i)
            s += alpha_[i] * y_[i] * kernel(X_[i], x);
        return s;
    }
};

// ─── SVM Multi-clase (one-vs-rest) ───────────────────────────
class SVM : public IClassifier {
public:
    SVM(double C = 1.0, double gamma = 1.0, int max_iter = 200, std::string kernel_type = "rbf")
        : C_(C), gamma_(gamma), max_iter_(max_iter), kernel_type_(kernel_type) {}

    void fit(const Dataset& train) override {
        labels_ = train.class_labels;
        svms_.clear();

        std::vector<std::vector<double>> X;
        for (auto& s : train.samples) X.push_back(s.x);

        for (int lbl : labels_) {
            std::vector<int> y;
            for (auto& s : train.samples)
                y.push_back(s.label == lbl ? 1 : -1);
            BinarySVM bsvm(C_, gamma_, max_iter_, kernel_type_);
            bsvm.fit(X, y);
            svms_.push_back(std::move(bsvm));
        }
    }

    // Clase con mayor valor de decisión (más "dentro" de su manifold)
    int predict(const std::vector<double>& x) const override {
        double best_d = -std::numeric_limits<double>::max();
        int    best_c = labels_[0];
        for (size_t k = 0; k < svms_.size(); ++k) {
            double d = svms_[k].decision_function(x);
            if (d > best_d) { best_d = d; best_c = labels_[k]; }
        }
        return best_c;
    }

    std::string name() const override {
        if (kernel_type_ == "linear") {
            return "SVM-Linear (C=" + std::to_string((int)C_) + ")";
        }
        return "SVM-RBF (C=" + std::to_string((int)C_)
             + " γ=" + std::to_string(gamma_).substr(0,4) + ")";
    }

private:
    double C_, gamma_;
    int    max_iter_;
    std::string kernel_type_;
    std::vector<int>       labels_;
    std::vector<BinarySVM> svms_;
};

} // namespace nsga2
