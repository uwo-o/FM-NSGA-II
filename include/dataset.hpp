#pragma once
// ============================================================
// include/dataset.hpp — Carga y gestión de datasets
// ============================================================
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <set>
#include <numeric>
#include "utils.hpp"

namespace nsga2 {

struct Sample {
    std::vector<double> x; // features
    int label;             // class label (entero, puede ser 0,1,2,...)
};

struct Dataset {
    std::vector<Sample> samples;
    int n_features   = 0;
    int n_classes    = 0;
    std::vector<int> class_labels; // etiquetas únicas ordenadas
    std::string name = "unnamed";

    bool empty()  const { return samples.empty(); }
    size_t size() const { return samples.size(); }

    // ─── Carga ───────────────────────────────────────────────
    // CSV: todas las columnas son features EXCEPTO la última, que es label (int)
    void load_csv(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + path);
        samples.clear();
        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<double> vals;
            while (std::getline(ss, tok, ',')) {
                try { vals.push_back(std::stod(tok)); }
                catch (...) { /* skip header token */ }
            }
            if (vals.size() < 2) continue;
            if (first) { n_features = (int)vals.size() - 1; first = false; }
            Sample s;
            s.x.assign(vals.begin(), vals.begin() + n_features);
            s.label = (int)std::round(vals.back());
            samples.push_back(std::move(s));
        }
        update_class_info();
    }

    // ─── Normalización min-max a [0,1] por feature ───────────
    void normalize() {
        if (samples.empty()) return;
        int d = n_features;
        std::vector<double> lo(d,  1e18), hi(d, -1e18);
        for (auto& s : samples)
            for (int i = 0; i < d; ++i) {
                lo[i] = std::min(lo[i], s.x[i]);
                hi[i] = std::max(hi[i], s.x[i]);
            }
        for (auto& s : samples)
            for (int i = 0; i < d; ++i) {
                double range = hi[i] - lo[i];
                s.x[i] = (range > 1e-12) ? (s.x[i] - lo[i]) / range : 0.0;
            }
    }

    // ─── Split train/test ─────────────────────────────────────
    std::pair<Dataset, Dataset> train_test_split(double test_ratio = 0.2,
                                                  unsigned seed    = 42) const {
        std::vector<int> idx(samples.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 g(seed);
        std::shuffle(idx.begin(), idx.end(), g);

        size_t n_test = (size_t)(samples.size() * test_ratio);
        Dataset train, test;
        train.name = name + "_train";
        test.name  = name + "_test";
        train.n_features = test.n_features = n_features;

        for (size_t i = 0; i < samples.size(); ++i) {
            if (i < n_test) test.samples.push_back(samples[idx[i]]);
            else            train.samples.push_back(samples[idx[i]]);
        }
        train.update_class_info();
        test.update_class_info();
        // test may lack some classes; use parent's info
        test.n_classes    = n_classes;
        test.class_labels = class_labels;
        return {train, test};
    }

    // ─── Conversión a binario (one-vs-rest) ───────────────────
    // Clase positiva → label 1, resto → label 0
    Dataset as_binary(int positive_class) const {
        Dataset d;
        d.name      = name + "_vs_" + std::to_string(positive_class);
        d.n_features = n_features;
        for (auto& s : samples) {
            Sample b;
            b.x     = s.x;
            b.label = (s.label == positive_class) ? 1 : 0;
            d.samples.push_back(b);
        }
        d.n_classes    = 2;
        d.class_labels = {0, 1};
        return d;
    }

    // ─── Conteos por clase ────────────────────────────────────
    std::vector<int> class_counts() const {
        std::vector<int> cnt(n_classes, 0);
        for (auto& s : samples) {
            auto it = std::find(class_labels.begin(), class_labels.end(), s.label);
            if (it != class_labels.end())
                ++cnt[it - class_labels.begin()];
        }
        return cnt;
    }

    // ─── Info ─────────────────────────────────────────────────
    void print_info() const {
        std::cout << "[Dataset] " << name
                  << "  samples=" << samples.size()
                  << "  features=" << n_features
                  << "  classes=" << n_classes << "\n";
        auto cnt = class_counts();
        for (int i = 0; i < n_classes; ++i)
            std::cout << "    class " << class_labels[i]
                      << ": " << cnt[i] << " samples\n";
    }

private:
    void update_class_info() {
        std::set<int> labels;
        for (auto& s : samples) labels.insert(s.label);
        class_labels.assign(labels.begin(), labels.end());
        n_classes = (int)class_labels.size();
    }
};

} // namespace nsga2
