// Header file for the Bagging function
// Yifan
// 03.11
#ifndef BAGGING_H
#define BAGGING_H

#include <vector>
#include <memory>
#include "../functions_tree/regression_tree.h"

/**
 * Bagging Class
 * This class implements the Bagging algorithm with multiple decision trees.
 */
class Bagging
{
public:
    /**
     * @brief Constructor
     * @param num_trees Specifies the number of decision trees to create
     * @param max_depth Specifies the maximum depth of each tree
     * @param criteria Pointer to the splitting criteria
     */
    Bagging(int num_trees, int max_depth, SplittingCriteria* criteria);

    /**
     * @brief Train the bagging model
     * @param data Feature matrix representing the features of each row
     * @param labels Vector of target values corresponding to each row
     */
    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);

    /**
     * @brief Predict for a single sample
     * @param sample A single feature vector
     * @return Prediction (average of predictions from multiple decision trees)
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
    int numTrees;  // Stores the number of decision trees
    int maxDepth;  // Stores the maximum depth of the decision trees
    SplittingCriteria* criteria;  // Pointer to the splitting criteria, used to determine splitting rules
    std::vector<std::unique_ptr<RegressionTree>> trees;  // Vector of smart pointers to store instances of all decision trees
    
    /**
     * Bootstrap a sample dataset (sampling with replacement)
     * @param data Original dataset's feature matrix
     * @param labels Original dataset's target vector
     * @return Pair of feature matrix and target vector of the sampled dataset
     */
    std::pair<std::vector<std::vector<double>>, std::vector<double>> bootstrapSample(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);
};

#endif
