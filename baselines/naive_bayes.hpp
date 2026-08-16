#pragma once
// ============================================================
// baselines/naive_bayes.hpp — Gaussian Naive Bayes
// ============================================================
#include <cmath>
#include <map>
#include <limits>
#include "classifier_base.hpp"

namespace nsga2 {

class GaussianNB : public IClassifier {
public:
    void fit(const Dataset& train) override {
        int d = train.n_features;
        // Calcular media y varianza por clase por feature
        for (int lbl : train.class_labels) {
            std::vector<double> sum(d, 0), sum2(d, 0);
            int cnt = 0;
            for (auto& s : train.samples)
                if (s.label == lbl) {
                    ++cnt;
                    for (int i = 0; i < d; ++i) {
                        sum[i]  += s.x[i];
                        sum2[i] += s.x[i] * s.x[i];
                    }
                }
            prior_[lbl] = static_cast<double>(cnt) / train.size();
            mean_[lbl].resize(d); var_[lbl].resize(d, 1e-6);
            for (int i = 0; i < d; ++i) {
                mean_[lbl][i] = sum[i] / cnt;
                double v = sum2[i]/cnt - mean_[lbl][i]*mean_[lbl][i];
                var_[lbl][i]  = std::max(v, 1e-6); // Laplace smoothing
            }
        }
        labels_ = train.class_labels;
    }

    int predict(const std::vector<double>& x) const override {
        double best_log = -std::numeric_limits<double>::max();
        int    best_lbl = labels_[0];
        for (int lbl : labels_) {
            double log_p = std::log(prior_.at(lbl));
            const auto& mu  = mean_.at(lbl);
            const auto& sig = var_.at(lbl);
            for (size_t i = 0; i < x.size(); ++i) {
                double diff = x[i] - mu[i];
                log_p -= 0.5 * (std::log(2*M_PI*sig[i]) + diff*diff/sig[i]);
            }
            if (log_p > best_log) { best_log = log_p; best_lbl = lbl; }
        }
        return best_lbl;
    }

    std::string name() const override { return "Gaussian Naive Bayes"; }

private:
    std::map<int, double>              prior_;
    std::map<int, std::vector<double>> mean_, var_;
    std::vector<int>                   labels_;
};

} // namespace nsga2
