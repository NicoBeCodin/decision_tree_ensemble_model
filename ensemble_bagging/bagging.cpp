#include "bagging.h"
#include <random>
#include <numeric>

Bagging::Bagging(int num_trees, int max_depth, SplittingCriteria* criteria)
    : numTrees(num_trees), maxDepth(max_depth), criteria(criteria) {
    // Initialize numTrees regression trees and add them to the trees vector
    for (int i = 0; i < numTrees; ++i) {
        trees.push_back(std::make_unique<RegressionTree>(maxDepth, criteria));
    }
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> 
Bagging::bootstrapSample(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    std::vector<std::vector<double>> sampled_data;  // Stores the sampled feature matrix
    std::vector<double> sampled_labels;             // Stores the sampled target values vector
    std::random_device rd;                          // Random device to seed the random engine
    std::default_random_engine generator(rd());     // Creates a random number generator with a random seed
    std::uniform_int_distribution<size_t> distribution(0, data.size() - 1);  // Uniform distribution in the range [0, data.size() - 1]

    // Perform bootstrap sampling for each sample
    for (size_t i = 0; i < data.size(); ++i) {
        size_t index = distribution(generator);     // Randomly select an index
        sampled_data.push_back(data[index]);        // Add the selected sample to sampled_data
        sampled_labels.push_back(labels[index]);    // Add the corresponding label to sampled_labels
    }

    return {sampled_data, sampled_labels};  // Return the sampled feature matrix and target values vector
}

void Bagging::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    // For each decision tree, generate a bootstrap sampled dataset and train the tree
    for (auto& tree : trees) {
        auto [sampled_data, sampled_labels] = bootstrapSample(data, labels);  // Generate sampled dataset
        tree->train(sampled_data, sampled_labels);                            // Train decision tree with the sampled dataset
    }
}

double Bagging::predict(const std::vector<double>& sample) {
    std::vector<double> predictions;  // Stores predictions from each tree for the sample
    for (const auto& tree : trees) {
        predictions.push_back(tree->predict(sample));  // Add each tree's prediction to predictions
    }
    // Calculate and return the average of all predictions
    return std::accumulate(predictions.begin(), predictions.end(), 0.0) / predictions.size();
}

/**
 * Evaluate model performance (calculate mean squared error on test set)
 */
double Bagging::evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) {
    double total_error = 0.0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        double prediction = predict(test_data[i]);
        total_error += std::pow(prediction - test_labels[i], 2);
    }
    return total_error / test_data.size();  // Return mean squared error
}
