#ifndef REGRESSION_TREE_H
#define REGRESSION_TREE_H

#include <vector>
#include <memory>
#include "splitting_criteria.h"

/**
 * RegressionTree class
 * Used for training, splitting, and predicting with a regression tree, managing nodes with smart pointers.
 */
class RegressionTree {
public:
    /**
     * Constructor
     * @param maxDepth The maximum depth of the tree
     * @param min_samples_split Minimum number of samples required to split a node
     * @param criteria Pointer to the splitting criteria, used to calculate the optimal split point
     */
    RegressionTree(int maxDepth, int min_samples_split, SplittingCriteria* criteria);

    /**
     * Train the regression tree
     * @param data Feature matrix, each row represents a sample
     * @param labels Target vector, representing the target value for each sample
     */
    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);
    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, const std::vector<double>& sampleWeights);

    /**
     * Predict for a single sample
     * @param sample Feature vector for a single sample
     * @return Predicted value
     */
    double predict(const std::vector<double>& sample) const;

    /**
     * Evaluate the model using a test set
     * @param test_data Test feature matrix
     * @param test_labels True target values of the test set
     * @return Mean Squared Error (MSE)
     */
    double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) const;

private:
    struct Node {
        int featureIndex;                 // Index of the splitting feature
        double threshold;                 // Splitting threshold
        double prediction;                // Prediction value for leaf nodes
        bool isLeaf;                      // Indicates whether this is a leaf node
        std::unique_ptr<Node> left;       // Left child node
        std::unique_ptr<Node> right;      // Right child node
        std::vector<int> data_indices;    // Indices of data samples associated with this node
    };

    std::unique_ptr<Node> root;           // Root node of the tree
    int maxDepth;                         // Maximum depth of the tree
    int min_samples_split;                // Minimum number of samples required to split
    SplittingCriteria* criteria;          // Splitting criteria for finding the best split

    /**
     * Recursive function for node splitting
     * @param node Current node
     * @param data Feature matrix
     * @param labels Target vector
     * @param depth Current depth
     */
    void splitNode(Node* node, const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int depth);
    void splitNode(Node* node, const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int depth, const std::vector<double>& sampleWeights);

    /**
     * Recursive function for making a prediction based on a sample and the current node
     * @param node The node to start the prediction from
     * @param sample Feature vector for a single sample
     * @return Predicted value
     */
    double predictNode(const Node* node, const std::vector<double>& sample) const;

    /**
     * Find the best splitting feature and threshold
     * @param data Feature matrix
     * @param labels Target vector
     * @param sampleWeights Sample weights for each data point
     * @return A pair of the optimal feature index and threshold
     */
    std::pair<int, double> findBestSplit(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, const std::vector<double>& sampleWeights);

    /**
     * Calculate the mean of the samples in the node
     * @param labels Target values of the current node
     * @return Prediction value for this node
     */
    double calculateMean(const std::vector<double>& labels);
    double calculateMean(const std::vector<double>& labels, const std::vector<double>& sampleWeights);

    /**
     * Calculate the Mean Squared Error (MSE) of the samples in the node
     * @param labels Target values of the current node
     * @return MSE value
     */
    double calculateMSE(const std::vector<double>& labels);
    double calculateMSE(const std::vector<double>& labels, const std::vector<double>& sampleWeights);
};

#endif // REGRESSION_TREE_H
