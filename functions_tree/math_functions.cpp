#include "math_functions.h"
#include <numeric>

/**
 * Calculate the mean of the samples
 */
double Math::calculateMean(const std::vector<double> &labels) 
{
    if (labels.empty())
    {
        return 0.0; // Return 0 if labels are empty, to prevent undefined behavior
    }
    double sum = std::accumulate(labels.begin(), labels.end(),0);
    return sum / labels.size();
}
double Math::calculateMeanWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int Idx : Indices) Sum += Labels[Idx];
    return Sum / Indices.size();
}

double Math::calculateMSEWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    double Mean = Math::calculateMeanWithIndices(Labels, Indices);
    double MSE = 0.0;
    for (int Idx : Indices) {
        double Value = Labels[Idx];
        MSE += std::pow(Value - Mean, 2);
    }
    return MSE / Indices.size();
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

double Math::calculateMedian(const std::vector<double>& values) {
    std::vector<double> sortedValues = values;
    std::sort(sortedValues.begin(), sortedValues.end());
    size_t n = sortedValues.size();
    if (n % 2 == 0) {
        return (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
    } else {
        return sortedValues[n / 2];
    }
}

double Math::calculateMAE(const std::vector<double>& values, double median) {
    double error = 0.0;
    for (double value : values) {
        error += std::abs(value - median);
    }
    return error / values.size();
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
        loss += std::pow(y_true[i]-y_pred[i], 2);
    }
    return loss / y_true.size();
}
