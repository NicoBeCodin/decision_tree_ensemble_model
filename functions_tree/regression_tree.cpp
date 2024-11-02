#include "regression_tree.h"
#include <limits>
#include <algorithm>
#include <cmath>

/**
 * Constructor
 */
RegressionTree::RegressionTree(int maxDepth, SplittingCriteria* criteria) 
    : maxDepth(maxDepth), criteria(criteria), root(nullptr) {}

/**
 * Train the regression tree
 */
void RegressionTree::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    root = std::make_unique<Node>();
    splitNode(root.get(), data, labels, 0);
}

/**
 * Predict for a single sample
 */
double RegressionTree::predict(const std::vector<double>& sample) {
    Node* currentNode = root.get();
    while (!currentNode->isLeaf) {
        if (sample[currentNode->featureIndex] <= currentNode->threshold) {
            currentNode = currentNode->left.get();
        } else {
            currentNode = currentNode->right.get();
        }
    }
    return currentNode->prediction;
}

/**
 * Evaluate the model using a test set
 */
double RegressionTree::evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) {
    double totalError = 0.0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        double prediction = predict(test_data[i]);
        totalError += std::pow(prediction - test_labels[i], 2);
    }
    return totalError / test_data.size();
}

/**
 * Recursive function: Node splitting
 */
void RegressionTree::splitNode(Node* node, const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int depth) {
    // Check stopping conditions
    if (depth >= maxDepth || labels.size() <= 1 || calculateMSE(labels) < 1e-6) {
        node->isLeaf = true;
        node->prediction = calculateMean(labels);
        return;
    }

    // Find the best split
    auto [bestFeature, bestThreshold] = findBestSplit(data, labels);
    if (bestFeature == -1) {  // If no valid split point is found
        node->isLeaf = true;
        node->prediction = calculateMean(labels);
        return;
    }

    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;

    // Split data
    std::vector<std::vector<double>> leftData, rightData;
    std::vector<double> leftLabels, rightLabels;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i][bestFeature] <= bestThreshold) {
            leftData.push_back(data[i]);
            leftLabels.push_back(labels[i]);
        } else {
            rightData.push_back(data[i]);
            rightLabels.push_back(labels[i]);
        }
    }

    // Recursively split the left and right child nodes
    node->left = std::make_unique<Node>();
    node->right = std::make_unique<Node>();
    splitNode(node->left.get(), leftData, leftLabels, depth + 1);
    splitNode(node->right.get(), rightData, rightLabels, depth + 1);
}

/**
 * Find the best splitting feature and threshold
 */
std::pair<int, double> RegressionTree::findBestSplit(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestMSE = std::numeric_limits<double>::max();

    for (int feature = 0; feature < data[0].size(); ++feature) {
        std::vector<double> featureValues;
        for (const auto& sample : data) {
            featureValues.push_back(sample[feature]);
        }
        std::sort(featureValues.begin(), featureValues.end());

        for (size_t i = 1; i < featureValues.size(); ++i) {
            double threshold = (featureValues[i - 1] + featureValues[i]) / 2;
            std::vector<double> leftLabels, rightLabels;
            for (size_t j = 0; j < data.size(); ++j) {
                if (data[j][feature] <= threshold) {
                    leftLabels.push_back(labels[j]);
                } else {
                    rightLabels.push_back(labels[j]);
                }
            }

            double mse = calculateMSE(leftLabels) * leftLabels.size() + calculateMSE(rightLabels) * rightLabels.size();
            if (mse < bestMSE) {
                bestMSE = mse;
                bestFeature = feature;
                bestThreshold = threshold;
            }
        }
    }
    return {bestFeature, bestThreshold};
}

/**
 * Calculate the mean of the samples
 */
double RegressionTree::calculateMean(const std::vector<double>& labels) {
    double sum = 0.0;
    for (double value : labels) sum += value;
    return sum / labels.size();
}

/**
 * Calculate the Mean Squared Error (MSE)
 */
double RegressionTree::calculateMSE(const std::vector<double>& labels) {
    double mean = calculateMean(labels);
    double mse = 0.0;
    for (double value : labels) mse += std::pow(value - mean, 2);
    return mse / labels.size();
}
