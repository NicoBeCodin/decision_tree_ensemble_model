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
    double sum = std::accumulate(labels.begin(), labels.end(),0);
    return sum / labels.size();
}
double Math::calculateMeanWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int Idx : Indices) Sum += Labels[Idx];
    return Sum / Indices.size();
}

double Math::calculateMedianWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    if (Indices.empty()) {
        throw std::invalid_argument("Indices cannot be empty for median calculation.");
    }

    // Extract the relevant labels
    std::vector<double> Subset;
    Subset.reserve(Indices.size());
    for (int Idx : Indices) {
        Subset.push_back(Labels[Idx]);
    }

    // Sort the subset to find the median
    std::sort(Subset.begin(), Subset.end());

    // Calculate the median
    size_t n = Subset.size();
    if (n % 2 == 0) {
        return (Subset[n / 2 - 1] + Subset[n / 2]) / 2.0; // Average of two middle elements
    } else {
        return Subset[n / 2]; // Middle element
    }
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

double Math::calculateMAEWithIndices(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    if (Indices.empty()) {
        return 0.0;
    }

    // Step 1: Extract target labels for this node
    std::vector<double> NodeLabels;
    for (int idx : Indices) {
        NodeLabels.push_back(Labels[idx]);
    }

    // Step 2: Calculate the median of labels (MAE uses the median)
    double Median = calculateMedian(NodeLabels);

    // Step 3: Calculate the Mean Absolute Error
    double MAE = calculateMAE(NodeLabels, Median);
    return MAE;
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
//NOT EFFICIENT
// double Math::calculateMedian(const std::vector<double>& values) {
//     std::vector<double> sortedValues = values;
//     std::sort(sortedValues.begin(), sortedValues.end());
//     size_t n = sortedValues.size();
//     if (n % 2 == 0) {
//         return (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
//     } else {
//         return sortedValues[n / 2];
//     }
// }

double Math::calculateMedian(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot compute median of an empty set.");
    }

    IncrementalMedian medianTracker;
    for (double val : values) {
        medianTracker.insert(val);
    }
    return medianTracker.getMedian();
}

double Math::incrementalMedian(std::vector<double>& sortedValues, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Cannot compute median of an empty subset.");
    }
    
    if (size % 2 == 1) { // Odd size
        return sortedValues[size / 2];
    } else { // Even size
        return (sortedValues[size / 2 - 1] + sortedValues[size / 2]) / 2.0;
    }
}



void Math::IncrementalMedian::insert(double value){
    if (leftMaxHeap.empty() || value <= leftMaxHeap.top()){
        leftMaxHeap.push(value);
    } else {
        rightMinHeap.push(value);
    }
    if (leftMaxHeap.size() > rightMinHeap.size()+ 1){
        rightMinHeap.push(leftMaxHeap.top());
        leftMaxHeap.pop();
        
    } else if(rightMinHeap.size() > leftMaxHeap.size()) {
        leftMaxHeap.push(rightMinHeap.top());
        rightMinHeap.pop();
    }
}

double Math::IncrementalMedian::getMedian() const {
    if (leftMaxHeap.size() > rightMinHeap.size()){
        return leftMaxHeap.top();
    } else {
        return (leftMaxHeap.top() + rightMinHeap.top())/2.0;
    }
}

double Math::calculateMedianSorted(const std::vector<double>& sortedValues) {
    size_t n = sortedValues.size();
    if (n % 2 == 0) {
        return (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
    } else {
        return sortedValues[n / 2];
    }
}


double Math::calculateMAE(const std::vector<double>& values, double mean) {
    double error = 0.0;
    for (double value : values) {
        error += std::abs(value - mean);
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

double Math::computeLossMSE(const std::vector<double> &y_true, const std::vector<double> &y_pred) 
{
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        loss += std::pow(y_true[i]-y_pred[i], 2);
    }
    return loss / y_true.size();
}

double Math::computeLossMAE(const std::vector<double> &y_true, const std::vector<double>& y_pred){
    double loss =0.0;
    for (size_t i = 0; i<y_true.size() ; ++i){
        loss+= std::abs(y_true[i]-y_pred[i]);
    }
    return loss /y_true.size();

}

