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
                   int max_depth, int min_samples_split, double min_impurity_decrease, int Criteria, int whichLossFunc)
    : n_estimators(n_estimators),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      min_impurity_decrease(min_impurity_decrease),
      learning_rate(learning_rate),
      loss_function(std::move(loss_function)),
      initial_prediction(0.0),
      Criteria(Criteria), 
      whichLossFunc(whichLossFunc) {
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
 * @param criteria MSE or MAE as a loss function (0 or 1)
 */
void Boosting::train(const std::vector<std::vector<double>>& X,
                     const std::vector<double>& y, int Criteria) {
    if (X.empty() || y.empty()) {
        return;
    }

    size_t n_samples = y.size();
    initializePrediction(y);
    std::vector<double> y_pred(n_samples, initial_prediction);

    // Training loop
    for (int i = 0; i < n_estimators; ++i) {
        // Calculate residuals (negative gradients)
        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

        // Create and train a new weak learner
        auto tree = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
        // Training with MSE only for the moment
        tree->train(X, residuals, Criteria);

        // Update predictions
        for (size_t j = 0; j < n_samples; ++j) {
            y_pred[j] += learning_rate * tree->predict(X[j]);
        }

        trees.push_back(std::move(tree));

       
        double current_loss = loss_function->computeLoss(y, y_pred);
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
         << initial_prediction << " "
         << Criteria << " "
         << whichLossFunc << "\n";
    
    // Sauvegarder chaque arbre avec un nom unique
    for (size_t i = 0; i < trees.size(); ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i]->saveTree(tree_filename);
    }
    
    file.close();
}

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
         >> initial_prediction
         >> Criteria
         >> whichLossFunc;
    
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

// Retourne les paramètres d'entraînement sous forme de dictionnaire (clé-valeur)
std::map<std::string, std::string> Boosting::getTrainingParameters() const {
    std::map<std::string, std::string> parameters;
    parameters["NumEstimators"] = std::to_string(n_estimators);
    parameters["LearningRate"] = std::to_string(learning_rate);
    parameters["MaxDepth"] = std::to_string(max_depth);
    parameters["MinSamplesSplit"] = std::to_string(min_samples_split);
    parameters["MinImpurityDecrease"] = std::to_string(min_impurity_decrease);
    parameters["InitialPrediction"] = std::to_string(initial_prediction);
    parameters["Criteria"] = std::to_string(Criteria);
    parameters["WhichLossFunction"] = std::to_string(whichLossFunc);
    return parameters;
}

// Retourne les paramètres d'entraînement sous forme d'une chaîne de caractères lisible
std::string Boosting::getTrainingParametersString() const {
    std::ostringstream oss;
    oss << "Training Parameters:\n";
    oss << "  - Number of Estimators: " << n_estimators << "\n";
    oss << "  - Learning Rate: " << learning_rate << "\n";
    oss << "  - Max Depth: " << max_depth << "\n";
    oss << "  - Min Samples Split: " << min_samples_split << "\n";
    oss << "  - Min Impurity Decrease: " << min_impurity_decrease << "\n";
    oss << "  - Initial Prediction: " << initial_prediction << "\n";
    oss << "  - Criteria: " << (Criteria == 0 ? "MSE" : "MAE") << "\n";
    oss << "  - Loss Function: " << (whichLossFunc == 0 ? "Least Squares Loss" : "Mean Absolute Loss") << "\n";
    return oss.str();
}
