#include "boosting.h"
#include <numeric>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>


/**
 * @brief Constructeur pour initialiser le modèle de Boosting
 * @param n_estimators Nombre de faibles apprenants (arbres de décision)
 * @param max_depth Profondeur maximale pour chaque arbre
 * @param learning_rate Taux d'apprentissage
 * @param criteria Critère de division
 * @param loss_function Fonction de perte (pour calculer le gradient et la perte)
 */
Boosting::Boosting(int n_estimators, double learning_rate,
                   std::unique_ptr<LossFunction> loss_function,
                   int max_depth, int min_samples_split, double min_impurity_decrease)
    : n_estimators(n_estimators),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      min_impurity_decrease(min_impurity_decrease),
      learning_rate(learning_rate),
      loss_function(std::move(loss_function)),
      initial_prediction(0.0) {
    trees.reserve(n_estimators);
}

/**
 * @brief Initialiser la prédiction initiale avec la moyenne des valeurs y
 * @param y Vecteur des étiquettes cibles
 */
void Boosting::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}


/**
 * @brief Entraîner le modèle de Boosting
 * @param X Matrice des caractéristiques
 * @param y Vecteur des étiquettes cibles
 */
void Boosting::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
    if (data.empty() || labels.empty()) {
        return;
    }

    // Initialize predictions with zeros
    std::vector<double> predictions(data.size(), 0.0);

    // Training loop
    for (int i = 0; i < n_estimators; ++i) {
        // Calculate residuals (negative gradients)
        std::vector<double> residuals = loss_function->negativeGradient(labels, predictions);

        // Create and train a new weak learner
        auto tree = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
        tree->train(data, residuals, 0);
        trees.push_back(std::move(tree));

        // Update predictions
        for (size_t j = 0; j < data.size(); ++j) {
            predictions[j] += learning_rate * trees.back()->predict(data[j]);
        }

        // Calculate and store the current loss
        double current_loss = loss_function->computeLoss(labels, predictions);
        
        #ifndef TESTING
        std::cout << "Iteration " << i + 1 << ", Loss: " << current_loss << std::endl;
        #endif
    }
}

/**
 * @brief Prédire pour un seul échantillon
 * @param x Vecteur des caractéristiques d'un échantillon
 * @return Prédiction pour l'échantillon
 */
double Boosting::predict(const std::vector<double>& x) const {
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
std::vector<double> Boosting::predict(const std::vector<std::vector<double>>& X) const {
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
double Boosting::evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const {
    std::vector<double> y_pred = predict(X_test);
    return loss_function->computeLoss(y_test, y_pred);
}

/**
 * @brief Sauvegarder le modèle de Boosting dans un fichier
 * @param filename Nom du fichier de sauvegarde
 */
void Boosting::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Sauvegarder tous les paramètres du modèle
    file << n_estimators << " " 
         << learning_rate << " " 
         << max_depth << " " 
         << min_samples_split << " " 
         << min_impurity_decrease << " "
         << initial_prediction << "\n";
    
    // Sauvegarder chaque arbre avec un nom unique
    for (size_t i = 0; i < trees.size(); ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i]->saveTree(tree_filename);
    }
    
    file.close();
}

/**
 * @brief Charger un modèle de Boosting à partir d'un fichier
 * @param filename Nom du fichier contenant le modèle sauvegardé
 */
void Boosting::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Charger tous les paramètres du modèle
    file >> n_estimators 
         >> learning_rate 
         >> max_depth 
         >> min_samples_split 
         >> min_impurity_decrease
         >> initial_prediction;
    
    // Réinitialiser et recharger les arbres
    trees.clear();
    trees.resize(n_estimators);
    
    // Charger chaque arbre
    for (int i = 0; i < n_estimators; ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i] = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
        trees[i]->loadTree(tree_filename);
    }
    
    file.close();
}
