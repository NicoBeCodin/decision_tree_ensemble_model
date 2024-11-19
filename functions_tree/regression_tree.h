#ifndef REGRESSION_TREE_H
#define REGRESSION_TREE_H

#include <vector>
#include <memory>
#include "splitting_criteria.h"
#include "math_functions.h"

/**
 * RegressionTree class
 * Used for training, splitting, and predicting with a regression tree, managing nodes with smart pointers.
 */
class RegressionTree: public Math {
public:
    /**
     * Constructor
     * @param maxDepth The maximum depth of the tree
     * @param criteria Pointer to the splitting criteria, used to calculate the optimal split point
     */
    RegressionTree(int maxDepth, SplittingCriteria* criteria);

    /**
     * Train the regression tree
     * @param data Feature matrix, each row represents a sample
     * @param labels Target vector, representing the target value for each sample
     */
    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);

    /**
     * Predict for a single sample
     * @param sample Feature vector for a single sample
     * @return Predicted value
     */
    double predict(const std::vector<double>& sample);

    /**
     * Evaluate the model using a test set
     * @param test_data Test feature matrix
     * @param test_labels True target values of the test set
     * @return Mean Squared Error (MSE)
     */
    double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels);

private:
    struct Node {
        int featureIndex;                // Index of the splitting feature
        double threshold;                // Splitting threshold
        double prediction;               // Prediction value for leaf nodes
        bool isLeaf;                     // Whether this is a leaf node
        std::unique_ptr<Node> left;      // Left child node
        std::unique_ptr<Node> right;     // Right child node
    };

    std::unique_ptr<Node> root;          // Root node
    int maxDepth;                        // Maximum depth of the tree
    SplittingCriteria* criteria;         // Splitting criteria

    /**
     * Recursive function for node splitting
     * @param node Current node
     * @param data Feature matrix
     * @param labels Target vector
     * @param depth Current depth
     */
    void splitNode(Node* node, const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int depth);

    /**
     * Find the best splitting feature and threshold
     * @param data Feature matrix
     * @param labels Target vector
     * @return A pair of the optimal feature and threshold
     */
    std::pair<int, double> findBestSplit(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);

};

#endif // REGRESSION_TREE_H
