#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H
#include <vector>
#include <cmath>


class Math{
public:
    /**
     * Calculate the mean of the samples in the node
     * @param labels Target values of the current node
     * @return Prediction value for this node
     */
    static double calculateMean (const std::vector<double>& labels) ;

    static double calculateStdDev(const std::vector<double>& data, double mean)  ;
    
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