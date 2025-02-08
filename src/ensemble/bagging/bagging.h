#ifndef BAGGING_H
#define BAGGING_H
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <fstream>
#include <iostream>

#include "../../functions/tree/decision_tree_single.h"
#include "../../functions/loss/loss_function.h"

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
    Bagging(int num_trees, int max_depth, int min_samples_split, double min_impurity_decrease, std::unique_ptr<LossFunction> loss_function = std::unique_ptr<LeastSquaresLoss>(), int Criteria = 0, int whichLossFunc = 0, int numThreads = 1);

    /**
     * @brief Train the Bagging model
     * @param data Feature matrix
     * @param labels Target vector
     */
    void train(const std::vector<double>& data, int rowLength, const std::vector<double>& labels, int criteria);

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
     * @return (Loss metric MSE or MAE)
     */
    double evaluate(const std::vector<double>& test_data, int rowLength,  const std::vector<double>& test_labels) const;

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

    // Retourne les paramètres d'entraînement sous forme de dictionnaire (clé-valeur)
    std::map<std::string, std::string> getTrainingParameters() const;
    // Retourne les paramètres d'entraînement sous forme d'une chaîne de caractères lisible
    std::string getTrainingParametersString() const;

private:
    int numTrees;  // Number of trees in the ensemble
    int maxDepth;  // Maximum depth of each tree
    int minSamplesSplit; // Minimum number of samples to split a node
    int Criteria;
    int whichLossFunc;
    double minImpurityDecrease; // Minimum impurity decrease for splitting
    std::unique_ptr<LossFunction> loss_function; //Function to calculate loss
    std::vector<std::unique_ptr<DecisionTreeSingle>> trees;  // Ensemble of decision trees

    int numThreads = 1; //Number of threads to use

    /**
     * @brief Bootstrap a sample dataset (sampling with replacement)
     * @param data Original dataset's feature matrix
     * @param labels Original dataset's target vector
     * @param sampled_data Output parameter for the sampled feature matrix
     * @param sampled_labels Output parameter for the sampled target vector
     */
    void bootstrapSample(const std::vector<double>& data, int rowLength, const std::vector<double>& labels,
                         std::vector<double>& sampled_data, std::vector<double>& sampled_labels);

    friend class FeatureImportance;
};

#endif
