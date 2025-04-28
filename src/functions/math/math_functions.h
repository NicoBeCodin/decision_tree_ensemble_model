#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <exception>
#include <stdexcept>
#include <queue>

class Math{
public:

    static double calculateMean(const std::vector<double>& labels) ;

    static double calculateMeanWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateMedianWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateMSEWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateStdDev(const std::vector<double>& data, double mean)  ;
    
    static double calculateMedian(const std::vector<double>& values);

    static double calculateMedianSorted(const std::vector<double>& sortedValues);

    static double calculateMAEWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices);

    static double calculateMAE(const std::vector<double>& values, double median);

    static double calculateMSE(const std::vector<double>& labels);

    static std::vector<double> negativeGradient(const std::vector<double>& y_true, const std::vector<double>& y_pred) ;

    static double computeLossMSE(const std::vector<double>& y_true, const std::vector<double>& y_pred) ;    

    static double computeLossMAE(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    static double incrementalMedian(std::vector<double>& sortedValues, size_t size);
    private:
    //More optimal way of calculting the median
    class IncrementalMedian {
        private:
        std::priority_queue<double> leftMaxHeap;
        std::priority_queue<double, std::vector<double>, std::greater<double>> rightMinHeap;
        
        public:
        void insert(double value);
        double getMedian() const;
    };
};
#endif