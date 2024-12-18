#ifndef BAGGING_H
#define BAGGING_H

#include "../functions_tree/decision_tree_single.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>

/**
 * Bagging Class
 * Implements the Bagging algorithm using multiple decision trees
 */
class Bagging
{
public:
    /**
     * @brief Constructor
     * @param num_trees Number of trees in the ensemble
     * @param max_depth Maximum depth of each tree
     * @param min_samples_split Minimum number of samples required to split a node
     * @param min_impurity_decrease Minimum impurity decrease required for a split
     */
    Bagging(int num_trees, int max_depth, int min_samples_split, double min_impurity_decrease);

    /**
     * @brief Train the Bagging model
     * @param data Feature matrix
     * @param labels Target vector
     */
    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels);

    /**
     * @brief Predict the target value for a single sample
     * @param sample A single feature vector
     * @return Prediction from the ensemble (average of all trees)
     */
    double predict(const std::vector<double>& sample) const;

    /**
     * @brief Evaluate the model performance on a test set
     * @param test_data Test feature matrix
     * @param test_labels Test target vector
     * @return Mean Squared Error (MSE)
     */
    double evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) const;

    /**
     * @brief Get the trees in the ensemble
     * @return A vector of pointers to the trees in the ensemble
     */
    const std::vector<std::unique_ptr<DecisionTreeSingle>>& getTrees() const { return trees; }

    /**
     * @brief Save the Bagging model to a file
     * @param filename The filename to save the model to
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load the Bagging model from a file
     * @param filename The filename to load the model from
     */
    void load(const std::string& filename);

private:
    int numTrees;  // Number of trees in the ensemble
    int maxDepth;  // Maximum depth of each tree
    int minSamplesSplit; // Minimum number of samples to split a node
    double minImpurityDecrease; // Minimum impurity decrease for splitting
    std::vector<std::unique_ptr<DecisionTreeSingle>> trees;  // Ensemble of decision trees

    /**
     * @brief Bootstrap a sample dataset (sampling with replacement)
     * @param data Original dataset's feature matrix
     * @param labels Original dataset's target vector
     * @param sampled_data Output parameter for the sampled feature matrix
     * @param sampled_labels Output parameter for the sampled target vector
     */
    void bootstrapSample(const std::vector<std::vector<double>>& data, const std::vector<double>& labels,
                         std::vector<std::vector<double>>& sampled_data, std::vector<double>& sampled_labels);

    friend class FeatureImportance;
};

#endif
