#ifndef SPLITTING_CRITERIA_H
#define SPLITTING_CRITERIA_H

#include <vector>
#include <numeric>
#include <cmath>
#include "math_functions.h"

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
class MeanSquaredError : public SplittingCriteria, public Math {
public:
    double calculate(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int featureIndex) override {
    //We can just call calculateMSE
    return Math::calculateMSE(labels);
    }
};

#endif // SPLITTING_CRITERIA_H
