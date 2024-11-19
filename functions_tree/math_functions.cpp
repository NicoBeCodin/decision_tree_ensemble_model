#include "math_functions.h"

/**
 * Calculate the mean of the samples
 */
double Math::calculateMean(const std::vector<double> &labels) 
{
    if (labels.empty())
    {
        return 0.0; // Return 0 if labels are empty, to prevent undefined behavior
    }
    double sum = 0.0;
    for (double value : labels)
        sum += value;
    return sum / labels.size();
}
//Takes also mean as parameter for optimization in data_clean.cpp
double Math::calculateStdDev(const std::vector<double>& data, double mean) {
        double sum = 0.0;
    for (const auto& value : data) {
        sum += std::pow(value - mean, 2);
    }
    return std::sqrt(sum / data.size());
}

/**
 * Calculate the Mean Squared Error (MSE)
 */
double Math::calculateMSE(const std::vector<double> &labels)
{
    if (labels.empty())
    {
        return 0.0; // Return 0 to handle empty label case, preventing division by zero
    }
    double mean = calculateMean(labels);
    double mse = 0.0;
    for (double value : labels)
        mse += std::pow(value - mean, 2);
    return mse / labels.size();
}

// Loss functions
std::vector<double> Math::negativeGradient(const std::vector<double> &y_true,
                                           const std::vector<double> &y_pred) 
{
    std::vector<double> residuals(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        residuals[i] = y_true[i] - y_pred[i];
    }
    return residuals;
}

double Math::computeLoss(const std::vector<double> &y_true, const std::vector<double> &y_pred) 
{
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        double diff = y_true[i] - y_pred[i];
        loss += diff * diff;
    }
    return loss / y_true.size();
}
