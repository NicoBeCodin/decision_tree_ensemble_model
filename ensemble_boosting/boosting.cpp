#include "boosting.h"
#include <numeric>
#include <cmath>
#include <iostream>

Boosting::Boosting(int num_trees, int max_depth, SplittingCriteria* criteria)
    : numTrees(num_trees), maxDepth(max_depth), criteria(criteria) {}

/**
 * Train the boosting model using weighted samples
 */
void Boosting::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    std::vector<double> sampleWeights(data.size(), 1.0 / data.size());  // Initialize sample weights

    for (int i = 0; i < numTrees; ++i) {
        // Train a tree on weighted data
        auto tree = std::make_unique<RegressionTree>(maxDepth, criteria);
        tree->train(data, labels, sampleWeights);
       
        // Compute predictions and calculate error
        double error = 0.0;
        for (size_t j = 0; j < data.size(); ++j) {
            double prediction = tree->predict(data[j]);
            error += sampleWeights[j] * std::pow(prediction - labels[j], 2);
        }

        // Calculate tree weight and update sample weights
        double treeWeight = 0.5 * std::log((1.0 - error) / error);
        treeWeights.push_back(treeWeight);
        trees.push_back(std::move(tree));

        // Update sample weights
        for (size_t j = 0; j < data.size(); ++j) {
            double prediction = trees.back()->predict(data[j]);
            sampleWeights[j] *= std::exp(-treeWeight * labels[j] * prediction);
        }

        // Normalize sample weights
        double sumWeights = std::accumulate(sampleWeights.begin(), sampleWeights.end(), 0.0);
        for (auto& weight : sampleWeights) {
            weight /= sumWeights;
        }
    }
}

/**
 * Predict for a single sample
 */
double Boosting::predict(const std::vector<double>& sample) {
    double weightedSum = 0.0;
    double totalWeight = 0.0;

    for (size_t i = 0; i < trees.size(); ++i) {
        weightedSum += treeWeights[i] * trees[i]->predict(sample);
        totalWeight += treeWeights[i];
    }

    return weightedSum / totalWeight;  // Return the weighted average prediction
}

/**
 * Evaluate model performance (calculate mean squared error on test set)
 */
double Boosting::evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) {
    double total_error = 0.0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        double prediction = predict(test_data[i]);
        total_error += std::pow(prediction - test_labels[i], 2);
    }
    return total_error / test_data.size();  // Return mean squared error
}
