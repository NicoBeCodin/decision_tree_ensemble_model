#include "boosting_XGBoost.h"
#include <numeric>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>

/**
 * @brief Constructeur pour initialiser le modèle XGBoost pour le boosting
 * @param n_estimators Nombre de faibles apprenants (arbres de décision)
 * @param max_depth Profondeur maximale pour chaque arbre
 * @param learning_rate Taux d'apprentissage
 * @param lambda Paramètre de régularisation L2
 * @param alpha Paramètre de régularisation L1
 * @param loss_function Fonction de perte (pour calculer le gradient et la perte)
 */
XGBoost::XGBoost(int n_estimators, int max_depth, double learning_rate, double lambda, double alpha,
                 std::unique_ptr<LossFunction> loss_function)
    : n_estimators(n_estimators), max_depth(max_depth), learning_rate(learning_rate),
      lambda(lambda), alpha(alpha), loss_function(std::move(loss_function)), initial_prediction(0.0) {
    trees.reserve(n_estimators);
}

/**
 * @brief Initialisation de la prédiction initiale avec la moyenne des valeurs y.
 * @param y Vecteur des étiquettes cibles)
 */
void XGBoost::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

/**
 * @brief Entraîner le modèle de Boosting
 * @param X Matrice des caractéristiques
 * @param y Vecteur des étiquettes cibles
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

        // Initialisation d'un nouvel arbre
        auto tree = std::make_unique<DecisionTreeXGBoost>(max_depth, 1, lambda, alpha);
        tree->train(X, residuals, y_pred);

        // Mise à jour des prédictions
        for (size_t j = 0; j < n_samples; ++j) {
            y_pred[j] += learning_rate * tree->predict(X[j]);
        }

        trees.push_back(std::move(tree));

        double loss = loss_function->computeLoss(y, y_pred);
        std::cout << "Iteration " << i + 1 << ", Loss: " << loss << std::endl;
    }
}

/**
 * @brief Prédire pour un seul échantillon
 * @param x Vecteur des caractéristiques d'un échantillon
 * @return Prédiction pour l'échantillon
 */
double XGBoost::predict(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& tree : trees) {
        y_pred += learning_rate * tree->predict(x);
    }
    return y_pred;
}

/**
 * @brief Prédire pour plusieurs échantillons
 * @param X Matrice des caractéristiques
 * @return Vecteur des prédictions pour chaque échantillon
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
 * @brief Évaluer la performance du modèle sur un ensemble de test
 * @param X_test Matrice des caractéristiques de test
 * @param y_test Vecteur des étiquettes cibles de test
 * @return Erreur quadratique moyenne (MSE)
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
    
    // Sauvegarder tous les paramètres du modèle
    file << n_estimators << " " 
         << max_depth << " " 
         << learning_rate << " "
         << lambda << " " 
         << alpha << " " 
         << initial_prediction << "\n";
    
    // Sauvegarder chaque arbre avec un nom unique
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
    
    // Charger tous les paramètres du modèle
    file >> n_estimators 
         >> max_depth 
         >> learning_rate 
         >> lambda 
         >> alpha 
         >> initial_prediction;
    
    // Réinitialiser et recharger les arbres
    trees.clear();
    trees.resize(n_estimators);
    
    // Charger chaque arbre
    for (int i = 0; i < n_estimators; ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i] = std::make_unique<DecisionTreeXGBoost>(max_depth, 1, lambda, alpha);
        trees[i]->loadTree(tree_filename);
    }
    
    file.close();
}

std::map<std::string, double> XGBoost::featureImportance(const std::vector<std::string>& feature_names) const {
    std::map<int, double> importance_scores;
    
    // Calculer l'importance pour chaque arbre
    for (const auto& tree : trees) {
        auto tree_importance = tree->getFeatureImportance();
        for (const auto& [feature, score] : tree_importance) {
            importance_scores[feature] += score;
        }
    }
    
    // Normaliser les scores
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