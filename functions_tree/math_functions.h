#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H
#include <vector>
#include <cmath>
#include <algorithm>


class Math{
public:
    /**
     * Calculate the mean of the samples in the node
     * @param labels Target values of the current node
     * @return Prediction value for this node
     */
    static double calculateMean (const std::vector<double>& labels) ;

    static double calculateMeanWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateMSEWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateStdDev(const std::vector<double>& data, double mean)  ;
    
    static double calculateMedian(const std::vector<double>& values);

    static double calculateMedianSorted(const std::vector<double>& sortedValues);

    static double calculateMAEWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateMAE(const std::vector<double>& values, double median);
    /**
     * Calculate the Mean Squared Error (MSE) of the samples in the node
     * @param labels Target values of the current node
     * @return MSE value
     */
    static double calculateMSE(const std::vector<double>& labels);

    static std::vector<double> negativeGradient(const std::vector<double>& y_true, const std::vector<double>& y_pred) ;

    static double computeLoss(const std::vector<double>& y_true, const std::vector<double>& y_pred) ;    
};

#endif