#ifndef BOOSTING_H
#define BOOSTING_H

#include <vector>
#include <memory>
#include "../functions_tree/regression_tree.h"

/**
 * Boosting Class
 * This class implements the Boosting algorithm with multiple decision trees.
 */
class Boosting {
public:
    /**
     * @brief Constructor
     * @param num_trees Specifies the number of decision trees to create
     * @param max_depth Specifies the maximum depth of each tree
     * @param criteria Pointer to the splitting criteria
     */
    Boosting(int num_trees, int max_depth, SplittingCriteria* criteria);

    /**
     * @brief Train the boosting model
     * @param data Feature matrix representing the features of each row
     * @param labels Vector of target values corresponding to each row
     */
    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);

    /**
     * @brief Predict for a single sample
     * @param sample A single feature vector
     * @return Prediction (weighted sum of predictions from multiple decision trees)
     */
    double predict(const std::vector<double>& sample);

    /**
     * @brief Evaluate model performance (calculates mean squared error on test set)
     * @param test_data Feature matrix of the test set
     * @param test_labels Vector of target values for the test set
     * @return Mean squared error
     */
    double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels);

private:
    int numTrees;  // Number of decision trees
    int maxDepth;  // Maximum depth of each tree
    SplittingCriteria* criteria;  // Pointer to the splitting criteria
    std::vector<std::unique_ptr<RegressionTree>> trees;  // Vector of regression trees
    std::vector<double> treeWeights;  // Weights for each tree based on performance
};

#endif
