#include "boosting_XGBoost.h"
#include <numeric>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>

/**
 * @brief Constructor to initialize the XGBoost model for boosting
 * @param n_estimators Number of weak learners (decision trees)
 * @param max_depth Maximum depth for each tree
 * @param learning_rate Learning rate
 * @param lambda L2 regularization parameter
 * @param alpha L1 regularization parameter
 * @param loss_function Loss function (to compute the gradient and loss)
 */
XGBoost::XGBoost(int n_estimators, int max_depth, double learning_rate, double lambda, double alpha,
                 std::unique_ptr<LossFunction> loss_function, int whichLossFunc)
    : n_estimators(n_estimators), max_depth(max_depth), learning_rate(learning_rate),
      lambda(lambda), alpha(alpha), loss_function(std::move(loss_function)), initial_prediction(0.0), whichLossFunc(whichLossFunc) {
    trees.reserve(n_estimators);
}

/**
 * @brief Initialize the initial prediction with the mean of the y values.
 * @param y Target labels vector
 */
void XGBoost::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

/**
 * @brief Train the Boosting model
 * @param X Feature matrix
 * @param y Target labels vector
 */
void XGBoost::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    size_t n_samples = y.size();
    initializePrediction(y);
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (int i = 0; i < n_estimators; ++i) {
        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

        // Regularization
        for (size_t j = 0; j < residuals.size(); ++j) {
            residuals[j] -= lambda * y_pred[j] + alpha * std::abs(y_pred[j]);
        }

        // Initialize a new tree
        auto tree = std::make_unique<DecisionTreeXGBoost>(max_depth, 1, lambda, alpha);
        tree->train(X, residuals, y_pred);

        // Update predictions
        for (size_t j = 0; j < n_samples; ++j) {
            y_pred[j] += learning_rate * tree->predict(X[j]);
        }

        trees.push_back(std::move(tree));

        double loss = loss_function->computeLoss(y, y_pred);
        std::cout << "Iteration " << i + 1 << ", Loss: " << loss << std::endl;
    }
}

/**
 * @brief Predict for a single sample
 * @param x Feature vector of a sample
 * @return Prediction for the sample
 */
double XGBoost::predict(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& tree : trees) {
        y_pred += learning_rate * tree->predict(x);
    }
    return y_pred;
}

/**
 * @brief Predict for multiple samples
 * @param X Feature matrix
 * @return Vector of predictions for each sample
 */
std::vector<double> XGBoost::predict(const std::vector<std::vector<double>>& X) const {
    size_t n_samples = X.size();
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (const auto& tree : trees) {
        for (size_t i = 0; i < n_samples; ++i) {
            y_pred[i] += learning_rate * tree->predict(X[i]);
        }
    }
    return y_pred;
}

/**
 * @brief Evaluate the model performance on a test set
 * @param X_test Test feature matrix
 * @param y_test Test target labels vector
 * @return Mean Squared Error (MSE)
 */
double XGBoost::evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const {
    std::vector<double> y_pred = predict(X_test);
    return loss_function->computeLoss(y_test, y_pred);
}

void XGBoost::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Save all model parameters
    file << n_estimators << " " 
         << max_depth << " " 
         << learning_rate << " "
         << lambda << " " 
         << alpha << " " 
         << initial_prediction << " "
         << whichLossFunc << "\n";
    
    // Save each tree with a unique name
    for (size_t i = 0; i < trees.size(); ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i]->saveTree(tree_filename);
    }
    
    file.close();
}

void XGBoost::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Load all model parameters
    file >> n_estimators
         >> max_depth 
         >> learning_rate 
         >> lambda 
         >> alpha 
         >> initial_prediction
         >> whichLossFunc;
    
    // Reset and reload trees
    trees.clear();
    trees.resize(n_estimators);
    
    // Load each tree
    for (int i = 0; i < n_estimators; ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i] = std::make_unique<DecisionTreeXGBoost>(max_depth, 1, lambda, alpha);
        trees[i]->loadTree(tree_filename);
    }
    
    file.close();
}

std::map<std::string, double> XGBoost::featureImportance(const std::vector<std::string>& feature_names) const {
    std::map<int, double> importance_scores;
    
    // Compute importance for each tree
    for (const auto& tree : trees) {
        auto tree_importance = tree->getFeatureImportance();
        for (const auto& [feature, score] : tree_importance) {
            importance_scores[feature] += score;
        }
    }
    
    // Normalize scores
    double total_importance = 0.0;
    for (const auto& [feature, score] : importance_scores) {
        total_importance += score;
    }
    
    std::map<std::string, double> normalized_importance;
    for (const auto& [feature, score] : importance_scores) {
        std::string feature_name;
        if (!feature_names.empty() && feature < static_cast<int>(feature_names.size())) {
            feature_name = feature_names[feature];
        } else {
            feature_name = "Feature " + std::to_string(feature);
        }
        normalized_importance[feature_name] = score / total_importance;
    }
    
    return normalized_importance;
}

// Retourne les paramètres d'entraînement sous forme de dictionnaire (clé-valeur)
std::map<std::string, std::string> XGBoost::getTrainingParameters() const {
    std::map<std::string, std::string> parameters;
    parameters["NumEstimators"] = std::to_string(n_estimators);
    parameters["MaxDepth"] = std::to_string(max_depth);
    parameters["LearningRate"] = std::to_string(learning_rate);
    parameters["Lambda"] = std::to_string(lambda);
    parameters["Alpha"] = std::to_string(alpha);
    parameters["InitialPrediction"] = std::to_string(initial_prediction);
    parameters["WhichLossFunction"] = std::to_string(whichLossFunc);
    return parameters;
}

// Retourne les paramètres d'entraînement sous forme d'une chaîne de caractères lisible
std::string XGBoost::getTrainingParametersString() const {
    std::ostringstream oss;
    oss << "Training Parameters:\n";
    oss << "  - Number of Estimators: " << n_estimators << "\n";
    oss << "  - Max Depth: " << max_depth << "\n";
    oss << "  - Learning Rate: " << learning_rate << "\n";
    oss << "  - Lambda (L2 Regularization): " << lambda << "\n";
    oss << "  - Alpha (L1 Regularization): " << alpha << "\n";
    oss << "  - Initial Prediction: " << initial_prediction << "\n";
    oss << "  - Loss Function: " << whichLossFunc << "\n";
    return oss.str();
}