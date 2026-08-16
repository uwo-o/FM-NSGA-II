#pragma once
// ============================================================
// baselines/classifier_base.hpp — Interfaz abstracta
// ============================================================
#include <vector>
#include <string>
#include "../include/dataset.hpp"

namespace nsga2 {

struct IClassifier {
    virtual ~IClassifier() = default;
    virtual void fit(const Dataset& train)               = 0;
    virtual int  predict(const std::vector<double>& x) const = 0;
    virtual std::string name() const                     = 0;

    std::vector<int> predict_all(const Dataset& data) const {
        std::vector<int> preds;
        preds.reserve(data.size());
        for (auto& s : data.samples)
            preds.push_back(predict(s.x));
        return preds;
    }
};

} // namespace nsga2
