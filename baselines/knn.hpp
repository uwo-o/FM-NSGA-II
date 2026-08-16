#pragma once
// ============================================================
// baselines/knn.hpp — k-Nearest Neighbors (brute force)
// ============================================================
#include <algorithm>
#include <map>
#include "classifier_base.hpp"
#include "../include/utils.hpp"

namespace nsga2 {

class KNN : public IClassifier {
public:
    explicit KNN(int k = 5) : k_(k) {}

    void fit(const Dataset& train) override {
        train_   = train.samples;
        k_eff_   = std::min(k_, (int)train_.size());
    }

    int predict(const std::vector<double>& x) const override {
        // Calcular distancias a todos los puntos de entrenamiento
        std::vector<std::pair<double, int>> dists;
        dists.reserve(train_.size());
        for (auto& s : train_)
            dists.emplace_back(dist2_sq(x, s.x), s.label);

        // k vecinos más cercanos
        std::partial_sort(dists.begin(), dists.begin() + k_eff_, dists.end());

        // Votación por mayoría
        std::map<int, int> votes;
        for (int i = 0; i < k_eff_; ++i) ++votes[dists[i].second];
        return std::max_element(votes.begin(), votes.end(),
            [](auto& a, auto& b){ return a.second < b.second; })->first;
    }

    std::string name() const override {
        return "k-NN (k=" + std::to_string(k_) + ")";
    }

private:
    int k_, k_eff_ = 5;
    std::vector<Sample> train_;
};

} // namespace nsga2
