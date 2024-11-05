#ifndef SPLITTING_CRITERIA_H
#define SPLITTING_CRITERIA_H

#include <vector>

/**
 * SplittingCriteria abstract base class, defines the interface for splitting criteria
 */
class SplittingCriteria {
public:
    virtual double calculate(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int featureIndex) = 0;
};

/**
 * MeanSquaredError class, inherits from SplittingCriteria and implements Mean Squared Error calculation, with a placeholder return value
 */
class MeanSquaredError : public SplittingCriteria {
public:
    double calculate(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int featureIndex) override {
        // The logic for calculating MSE should be implemented here
        return 0.0;  // Placeholder return value
    }
};

#endif // SPLITTING_CRITERIA_H
