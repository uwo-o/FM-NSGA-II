#pragma once
// ============================================================
// baselines/decision_tree.hpp — Árbol CART (Gini Impurity)
// ============================================================
// Implementa un árbol de decisión binario con:
//   - División por umbral en un feature (CART)
//   - Impureza de Gini para selección de splits
//   - max_depth y min_samples_split para regularización
//   - Soporte multi-clase nativo
// ============================================================
#include <map>
#include <limits>
#include <memory>
#include <cmath>
#include "classifier_base.hpp"

namespace nsga2 {

class DecisionTree : public IClassifier {
public:
    explicit DecisionTree(int max_depth = 10, int min_samples = 2)
        : max_depth_(max_depth), min_samples_(min_samples) {}

    void fit(const Dataset& train) override {
        labels_ = train.class_labels;
        root_   = build(train.samples, 0);
    }

    int predict(const std::vector<double>& x) const override {
        return traverse(root_.get(), x);
    }

    std::string name() const override {
        return "Decision Tree (depth=" + std::to_string(max_depth_) + ")";
    }

private:
    struct Node {
        bool is_leaf  = false;
        int  leaf_cls = 0;
        int  feat_idx = 0;
        double thresh = 0.0;
        std::unique_ptr<Node> left, right;
    };

    int max_depth_, min_samples_;
    std::vector<int>     labels_;
    std::unique_ptr<Node> root_;

    // ─── Impureza de Gini ─────────────────────────────────────
    static double gini(const std::vector<Sample>& samples) {
        if (samples.empty()) return 0.0;
        std::map<int, int> cnt;
        for (auto& s : samples) ++cnt[s.label];
        double imp = 1.0;
        double n   = (double)samples.size();
        for (auto& [lbl, c] : cnt) {
            double p = c / n;
            imp -= p * p;
        }
        return imp;
    }

    // ─── Clase mayoritaria ────────────────────────────────────
    static int majority_class(const std::vector<Sample>& samples) {
        std::map<int, int> cnt;
        for (auto& s : samples) ++cnt[s.label];
        return std::max_element(cnt.begin(), cnt.end(),
            [](auto& a, auto& b){ return a.second < b.second; })->first;
    }

    // ─── Construcción recursiva ───────────────────────────────
    std::unique_ptr<Node> build(const std::vector<Sample>& samples, int depth) {
        auto node = std::make_unique<Node>();
        if ((int)samples.size() < min_samples_ || depth >= max_depth_) {
            node->is_leaf  = true;
            node->leaf_cls = majority_class(samples);
            return node;
        }
        // Verificar si todos son de la misma clase
        bool pure = true;
        int first_lbl = samples[0].label;
        for (auto& s : samples) if (s.label != first_lbl) { pure = false; break; }
        if (pure) {
            node->is_leaf  = true;
            node->leaf_cls = first_lbl;
            return node;
        }

        // Encontrar el mejor split
        int    best_feat  = 0;
        double best_thresh = 0.0;
        double best_score  = std::numeric_limits<double>::max();
        int d = (int)samples[0].x.size();

        for (int feat = 0; feat < d; ++feat) {
            // Recoger valores únicos del feature
            std::vector<double> vals;
            for (auto& s : samples) vals.push_back(s.x[feat]);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

            for (size_t vi = 0; vi + 1 < vals.size(); ++vi) {
                double thresh = (vals[vi] + vals[vi+1]) / 2.0;
                std::vector<Sample> left, right;
                for (auto& s : samples)
                    (s.x[feat] <= thresh ? left : right).push_back(s);
                if (left.empty() || right.empty()) continue;
                double score = (left.size()  * gini(left)
                              + right.size() * gini(right)) / samples.size();
                if (score < best_score) {
                    best_score  = score;
                    best_feat   = feat;
                    best_thresh = thresh;
                }
            }
        }

        // Si no se encontró split válido → hoja
        if (best_score >= std::numeric_limits<double>::max()) {
            node->is_leaf  = true;
            node->leaf_cls = majority_class(samples);
            return node;
        }

        node->feat_idx = best_feat;
        node->thresh   = best_thresh;
        std::vector<Sample> left, right;
        for (auto& s : samples)
            (s.x[best_feat] <= best_thresh ? left : right).push_back(s);

        node->left  = build(left,  depth + 1);
        node->right = build(right, depth + 1);
        return node;
    }

    // ─── Predicción ───────────────────────────────────────────
    int traverse(const Node* node, const std::vector<double>& x) const {
        if (node->is_leaf) return node->leaf_cls;
        return (x[node->feat_idx] <= node->thresh)
               ? traverse(node->left.get(),  x)
               : traverse(node->right.get(), x);
    }
};

} // namespace nsga2
