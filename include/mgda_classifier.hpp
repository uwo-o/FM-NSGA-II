#pragma once
// ============================================================
// include/mgda_classifier.hpp — MGDA One-vs-Rest Classifier
// ============================================================
//
// Wrapper multiclase sobre mgda_train(). Cada clase instancia
// su propio AngularManifold y lo entrena vía MGDA independientemente.
// La clasificación sigue la misma lógica que Classifier<AngularManifold>:
//   predict(x) = argmin_k D_S(x, M_k)
// ============================================================
#include <iostream>
#include <vector>
#include "mgda.hpp"
#include "dataset.hpp"

namespace mgda {

class MGDAClassifier {
public:
    explicit MGDAClassifier(const MGDAConfig& cfg = {}) : cfg_(cfg) {}

    void fit(const nsga2::Dataset& data) {
        class_labels_ = data.class_labels;
        n_classes_    = (int)class_labels_.size();
        manifolds_.clear();
        trajectories_.clear();

        for (int k = 0; k < n_classes_; ++k) {
            int cls = class_labels_[k];
            std::cout << "\n[MGDA] Clase " << cls << " vs. resto...\n";

            // Dataset binario: +1 = clase k, -1 = el resto
            nsga2::Dataset binary;
            std::vector<double> pos_center(data.samples[0].x.size(), 0.0);
            int pos_count = 0;
            for (auto& s : data.samples) {
                nsga2::Sample bs;
                bs.x     = s.x;
                bs.label = (s.label == cls) ? 1 : -1;
                binary.samples.push_back(bs);
                if (s.label == cls) {
                    for (int d = 0; d < (int)s.x.size(); ++d)
                        pos_center[d] += s.x[d];
                    ++pos_count;
                }
            }
            binary.class_labels = {-1, 1};
            if (pos_count > 0)
                for (auto& c : pos_center) c /= pos_count;

            auto result = mgda_train(binary, pos_center, cfg_);
            manifolds_.push_back(result.manifold);
            trajectories_.push_back(result.trajectory);
        }
    }

    // Predicción: clase con menor distancia con signo al manifold
    int predict(const std::vector<double>& x) const {
        double best_d = 1e18;
        int    best_c = class_labels_[0];
        for (int k = 0; k < n_classes_; ++k) {
            double d = manifolds_[k].signed_distance(x);
            if (d < best_d) { best_d = d; best_c = class_labels_[k]; }
        }
        return best_c;
    }

    std::vector<int> predict_all(const nsga2::Dataset& data) const {
        std::vector<int> preds(data.size());
        for (int i = 0; i < data.size(); ++i)
            preds[i] = predict(data.samples[i].x);
        return preds;
    }

    const std::vector<nsga2::AngularManifold>&       manifolds()    const { return manifolds_;    }
    const std::vector<std::vector<TrajectoryPoint>>& trajectories() const { return trajectories_; }
    const std::vector<int>&                          labels()       const { return class_labels_;  }

private:
    MGDAConfig  cfg_;
    int         n_classes_ = 0;
    std::vector<int>                          class_labels_;
    std::vector<nsga2::AngularManifold>       manifolds_;
    std::vector<std::vector<TrajectoryPoint>> trajectories_;
};

} // namespace mgda
