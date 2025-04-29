#ifndef BOOSTING_ADVANCED_H
#define BOOSTING_ADVANCED_H
#include <vector>
#include "boosting_improved.h"

// Interface class for compatibility with existing code
class AdvancedGBDT {
public:
    // Constructor compatible with existing code
    AdvancedGBDT(int n_estimators, int max_depth, double learning_rate = 0.1,
        bool useDart = false, double drop_rate = 0.1, double skip_rate = 0.0)
        : impl(n_estimators, max_depth, learning_rate, useDart, drop_rate, skip_rate,
        ImprovedGBDT::BinningMethod::NONE, 0) {}

    // Training method
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        impl.fit(X, y);
    }

    // Prediction for multiple samples
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        return impl.predict(X);
    }

    // Prediction for a single sample
    double predict(const std::vector<double>& x) const {
        return impl.predict(x);
    }

private:
    ImprovedGBDT impl; // Using improved implementation
};

#endif // BOOSTING_ADVANCED_H