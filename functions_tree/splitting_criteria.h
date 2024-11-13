#ifndef SPLITTING_CRITERIA_H
#define SPLITTING_CRITERIA_H

#include <vector>
#include <numeric>
#include <cmath>

/**
 * SplittingCriteria abstract base class, defines the interface for splitting criteria
 */
class SplittingCriteria {
public:
    virtual double calculate(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int featureIndex) = 0;
};

/**
 * MeanSquaredError class, inherits from SplittingCriteria and implements Mean Squared Error calculation
 */
class MeanSquaredError : public SplittingCriteria {
public:
    double calculate(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int featureIndex) override {
        if (labels.empty()) {
            return 0.0;  // Return 0 if labels are empty to prevent undefined behavior
        }
        double mean = std::accumulate(labels.begin(), labels.end(), 0.0) / labels.size();
        double mse = 0.0;
        for (double value : labels) {
            mse += std::pow(value - mean, 2);
        }
        return mse / labels.size();
    }
};

#endif // SPLITTING_CRITERIA_H
