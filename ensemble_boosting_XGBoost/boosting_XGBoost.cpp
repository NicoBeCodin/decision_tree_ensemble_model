#include "boosting_XGBoost.h"
#include <numeric>
#include <cmath>
#include <iostream>

/**
 * @brief Constructeur d'un arbre de régression.
 */
RegressionTree::RegressionTree() : constant_prediction(0.0) {}

void RegressionTree::train(const std::vector<std::vector<double>>& X, const std::vector<double>& residuals) {
    constant_prediction = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
}

double RegressionTree::predict(const std::vector<double>& x) const {
    return constant_prediction;
}

/**
 * @brief Constructeur de XGBoost.
 */
XGBoost::XGBoost(int n_estimators, int max_depth, double learning_rate, double lambda, double alpha,
                 std::unique_ptr<LossFunction> loss_function)
    : n_estimators(n_estimators), max_depth(max_depth), learning_rate(learning_rate),
      lambda(lambda), alpha(alpha), loss_function(std::move(loss_function)), initial_prediction(0.0) {}

/**
 * @brief Initialisation de la prédiction initiale avec la moyenne des valeurs y.
 */
void XGBoost::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

/**
 * @brief Entraîner le modèle XGBoost.
 */
void XGBoost::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    size_t n_samples = y.size();
    initializePrediction(y);
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (int i = 0; i < n_estimators; ++i) {
        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

        // Régularisation
        for (size_t j = 0; j < residuals.size(); ++j) {
            residuals[j] -= lambda * y_pred[j] + alpha * std::abs(y_pred[j]);
        }

        auto tree = std::make_unique<RegressionTree>();
        tree->train(X, residuals);

        for (size_t j = 0; j < n_samples; ++j) {
            y_pred[j] += learning_rate * tree->predict(X[j]);
        }

        estimators.push_back(std::move(tree));

        double loss = loss_function->computeLoss(y, y_pred);
        std::cout << "Estimateur " << i + 1 << ", Perte: " << loss << std::endl;
    }
}

/**
 * @brief Prédire pour un seul échantillon.
 */
double XGBoost::predict(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& tree : estimators) {
        y_pred += learning_rate * tree->predict(x);
    }
    return y_pred;
}

/**
 * @brief Prédire pour plusieurs échantillons.
 */
std::vector<double> XGBoost::predict(const std::vector<std::vector<double>>& X) const {
    size_t n_samples = X.size();
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (const auto& tree : estimators) {
        for (size_t i = 0; i < n_samples; ++i) {
            y_pred[i] += learning_rate * tree->predict(X[i]);
        }
    }
    return y_pred;
}
