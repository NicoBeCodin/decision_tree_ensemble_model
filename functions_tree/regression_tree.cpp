#include "regression_tree.h"
#include <limits>
#include <algorithm>
#include <cmath>

/**
 * Constructor
 */
RegressionTree::RegressionTree(int maxDepth, int min_samples_split, SplittingCriteria* criteria) 
    : maxDepth(maxDepth), min_samples_split(min_samples_split), criteria(criteria), root(nullptr) {}

/**
 * Train the regression tree
 */
void RegressionTree::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    root = std::make_unique<Node>();
    splitNode(root.get(), data, labels, 0);
}

void RegressionTree::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, const std::vector<double>& sampleWeights) {
    root = std::make_unique<Node>();
    splitNode(root.get(), data, labels, 0, sampleWeights);
}

/**
 * Predict for a single sample
 */
double RegressionTree::predict(const std::vector<double>& sample) const {
    const Node* currentNode = root.get();
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
double RegressionTree::evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) const {
    double totalError = 0.0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        double prediction = predict(test_data[i]);
        totalError += std::pow(prediction - test_labels[i], 2);
    }
    return totalError / test_data.size();
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
 * Calculate the mean of the samples with weights
 */
double RegressionTree::calculateMean(const std::vector<double>& labels, const std::vector<double>& sampleWeights) {
    double weightedSum = 0.0;
    double weightTotal = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
        weightedSum += labels[i] * sampleWeights[i];
        weightTotal += sampleWeights[i];
    }
    return weightedSum / weightTotal;
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

/**
 * Calculate the Mean Squared Error (MSE) with weights
 */
double RegressionTree::calculateMSE(const std::vector<double>& labels, const std::vector<double>& sampleWeights) {
    double mean = calculateMean(labels, sampleWeights);
    double weightedMSE = 0.0;
    double weightTotal = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
        weightedMSE += sampleWeights[i] * std::pow(labels[i] - mean, 2);
        weightTotal += sampleWeights[i];
    }
    return weightedMSE / weightTotal;
}

/**
 * Recursive function: Node splitting with sample weights
 */
void RegressionTree::splitNode(Node* node, const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int depth, const std::vector<double>& sampleWeights) {
    if (depth >= maxDepth || labels.size() <= 1 || calculateMSE(labels, sampleWeights) < 1e-6) {
        node->isLeaf = true;
        node->prediction = calculateMean(labels, sampleWeights);
        return;
    }

    auto [bestFeature, bestThreshold] = findBestSplit(data, labels, sampleWeights);
    if (bestFeature == -1) {
        node->isLeaf = true;
        node->prediction = calculateMean(labels, sampleWeights);
        return;
    }

    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;

    std::vector<std::vector<double>> leftData, rightData;
    std::vector<double> leftLabels, rightLabels;
    std::vector<double> leftWeights, rightWeights;

    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i][bestFeature] <= bestThreshold) {
            leftData.push_back(data[i]);
            leftLabels.push_back(labels[i]);
            leftWeights.push_back(sampleWeights[i]);
        } else {
            rightData.push_back(data[i]);
            rightLabels.push_back(labels[i]);
            rightWeights.push_back(sampleWeights[i]);
        }
    }

    node->left = std::make_unique<Node>();
    node->right = std::make_unique<Node>();
    splitNode(node->left.get(), leftData, leftLabels, depth + 1, leftWeights);
    splitNode(node->right.get(), rightData, rightLabels, depth + 1, rightWeights);
}

/**
 * Find the best splitting feature and threshold with sample weights
 */
std::pair<int, double> RegressionTree::findBestSplit(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, const std::vector<double>& sampleWeights) {
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestWeightedMSE = std::numeric_limits<double>::max();

    for (int feature = 0; feature < data[0].size(); ++feature) {
        std::vector<double> featureValues;
        for (const auto& sample : data) {
            featureValues.push_back(sample[feature]);
        }
        std::sort(featureValues.begin(), featureValues.end());

        for (size_t i = 1; i < featureValues.size(); ++i) {
            double threshold = (featureValues[i - 1] + featureValues[i]) / 2;
            std::vector<double> leftLabels, rightLabels;
            std::vector<double> leftWeights, rightWeights;

            for (size_t j = 0; j < data.size(); ++j) {
                if (data[j][feature] <= threshold) {
                    leftLabels.push_back(labels[j]);
                    leftWeights.push_back(sampleWeights[j]);
                } else {
                    rightLabels.push_back(labels[j]);
                    rightWeights.push_back(sampleWeights[j]);
                }
            }

            double weightedMSE = calculateMSE(leftLabels, leftWeights) * leftLabels.size() +
                                 calculateMSE(rightLabels, rightWeights) * rightLabels.size();

            if (weightedMSE < bestWeightedMSE) {
                bestWeightedMSE = weightedMSE;
                bestFeature = feature;
                bestThreshold = threshold;
            }
        }
    }
    return {bestFeature, bestThreshold};
}
