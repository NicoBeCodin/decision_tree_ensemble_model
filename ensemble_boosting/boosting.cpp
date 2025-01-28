// #include "boosting.h"
// #include <numeric>
// #include <iostream>
// #include <random>
// #include <algorithm>
// #include <stdexcept>
// #include <fstream>

// /**
//  * @brief Constructeur pour initialiser le modèle de Boosting
//  * @param n_estimators Nombre de faibles apprenants (arbres de décision)
//  * @param max_depth Profondeur maximale pour chaque arbre
//  * @param learning_rate Taux d'apprentissage
//  * @param criteria Critère de division
//  * @param loss_function Fonction de perte (pour calculer le gradient et la perte)
//  */
// Boosting::Boosting(int n_estimators, double learning_rate,
//                    std::unique_ptr<LossFunction> loss_function,
//                    int max_depth, int min_samples_split, double min_impurity_decrease)
//     : n_estimators(n_estimators),
//       max_depth(max_depth),
//       min_samples_split(min_samples_split),
//       min_impurity_decrease(min_impurity_decrease),
//       learning_rate(learning_rate),
//       loss_function(std::move(loss_function)),
//       initial_prediction(0.0) {
//     trees.reserve(n_estimators);
// }

// /**
//  * @brief Initialiser la prédiction initiale avec la moyenne des valeurs y
//  * @param y Vecteur des étiquettes cibles
//  */
// void Boosting::initializePrediction(const std::vector<double>& y) {
//     initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
// }


// /**
//  * @brief Entraîner le modèle de Boosting
//  * @param X Matrice des caractéristiques
//  * @param y Vecteur des étiquettes cibles
//  * @param criteria MSE or MAE as a loss function (0 or 1)
//  */
// void Boosting::train(const std::vector<std::vector<double>>& X,
//                      const std::vector<double>& y, int criteria) {
//     if (X.empty() || y.empty()) {
//         return;
//     }

//     size_t n_samples = y.size();
//     initializePrediction(y);
//     std::vector<double> y_pred(n_samples, initial_prediction);

//     // Training loop
//     for (int i = 0; i < n_estimators; ++i) {
//         // Calculate residuals (negative gradients)
//         std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

//         // Create and train a new weak learner
//         auto tree = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
//         // Training with MSE only for the moment
//         tree->train(X, residuals, criteria);

//         // Update predictions
//         for (size_t j = 0; j < n_samples; ++j) {
//             y_pred[j] += learning_rate * tree->predict(X[j]);
//         }

//         trees.push_back(std::move(tree));

       
//         double current_loss = loss_function->computeLoss(y, y_pred);
//         #ifndef TESTING
//         std::cout << "Iteration " << i + 1 << ", Loss: " << current_loss << std::endl;
//         #endif    
//     }
// }

// /**
//  * @brief Prédire pour un seul échantillon
//  * @param x Vecteur des caractéristiques d'un échantillon
//  * @return Prédiction pour l'échantillon
//  */
// double Boosting::predict(const std::vector<double>& x) const {
//     double y_pred = initial_prediction;
//     for (const auto& tree : trees) {
//         y_pred += learning_rate * tree->predict(x);
//     }
//     return y_pred;
// }


// /**
//  * @brief Prédire pour plusieurs échantillons
//  * @param X Matrice des caractéristiques
//  * @return Vecteur des prédictions pour chaque échantillon
//  */
// std::vector<double> Boosting::predict(const std::vector<std::vector<double>>& X) const {
//     size_t n_samples = X.size();
//     std::vector<double> y_pred(n_samples, initial_prediction);
 
//     for (const auto& tree : trees) {
//         for (size_t i = 0; i < n_samples; ++i) {
//             y_pred[i] += learning_rate * tree->predict(X[i]);
//         }
//     }
//     return y_pred;
// }

// /**
//  * @brief Évaluer la performance du modèle sur un ensemble de test
//  * @param X_test Matrice des caractéristiques de test
//  * @param y_test Vecteur des étiquettes cibles de test
//  * @return Erreur quadratique moyenne (MSE)
//  */
// double Boosting::evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const {
//     std::vector<double> y_pred = predict(X_test);
//     return loss_function->computeLoss(y_test, y_pred);
// }

// void Boosting::save(const std::string& filename) const {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Cannot open file for writing: " + filename);
//     }
    
//     // Sauvegarder tous les paramètres du modèle
//     file << n_estimators << " " 
//          << learning_rate << " " 
//          << max_depth << " " 
//          << min_samples_split << " " 
//          << min_impurity_decrease << " "
//          << initial_prediction << "\n";
    
//     // Sauvegarder chaque arbre avec un nom unique
//     for (size_t i = 0; i < trees.size(); ++i) {
//         std::string tree_filename = filename + "_tree_" + std::to_string(i);
//         trees[i]->saveTree(tree_filename);
//     }
    
//     file.close();
// }

// void Boosting::load(const std::string& filename) {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Cannot open file for reading: " + filename);
//     }
    
//     // Charger tous les paramètres du modèle
//     file >> n_estimators 
//          >> learning_rate 
//          >> max_depth 
//          >> min_samples_split 
//          >> min_impurity_decrease
//          >> initial_prediction;
    
//     // Réinitialiser et recharger les arbres
//     trees.clear();
//     trees.resize(n_estimators);
    
//     // Charger chaque arbre
//     for (int i = 0; i < n_estimators; ++i) {
//         std::string tree_filename = filename + "_tree_" + std::to_string(i);
//         trees[i] = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
//         trees[i]->loadTree(tree_filename);
//     }
    
//     file.close();
// }


//LInearisation



#include "boosting.h"
#include <numeric>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>

/**
 * @brief Constructor to initialize the Boosting model
 * @param n_estimators Number of weak learners (decision trees)
 * @param max_depth Maximum depth for each tree
 * @param learning_rate Learning rate
 * @param loss_function Loss function to calculate gradients and errors
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
 * @brief Initialize the initial prediction with the mean of target values
 * @param y Vector of target labels
 */
void Boosting::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

/**
 * @brief Train the Boosting model
 * @param X Flattened feature matrix (1D vector)
 * @param rowLength Number of features in each sample
 * @param y Vector of target labels
 * @param criteria MSE or MAE as a loss function (0 or 1)
 */
void Boosting::train(const std::vector<double>& X, int rowLength,
                     const std::vector<double>& y, int criteria) {
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
        tree->train(X, rowLength, residuals, criteria);

        // Update predictions
        for (size_t j = 0; j < n_samples; ++j) {
            std::vector<double> sample(X.begin() + j * rowLength, X.begin() + (j + 1) * rowLength);
            y_pred[j] += learning_rate * tree->predict(sample);
        }

        trees.push_back(std::move(tree));

        // Calculate and display loss
        double current_loss = loss_function->computeLoss(y, y_pred);

        std::cout << "Iteration " << i + 1 << ", Loss: " << current_loss << std::endl;
    }
}

/**
 * @brief Predict for a single sample
 * @param x Vector of features for a single sample
 * @return Prediction for the sample
 */
double Boosting::predict_single(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& tree : trees) {
        y_pred += learning_rate * tree->predict(x);
    }
    return y_pred;
}

/**
 * @brief Predict for multiple samples
 * @param X Flattened feature matrix (1D vector)
 * @param rowLength Number of features in each sample
 * @return Vector of predictions for each sample
 */
std::vector<double> Boosting::predict(const std::vector<double>& X, int rowLength) const {
    size_t n_samples = X.size() / rowLength;
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (const auto& tree : trees) {
        for (size_t i = 0; i < n_samples; ++i) {
            std::vector<double> sample(X.begin() + i * rowLength, X.begin() + (i + 1) * rowLength);
            y_pred[i] += learning_rate * tree->predict(sample);
        }
    }
    return y_pred;
}

/**
 * @brief Evaluate the model's performance on a test set
 * @param X_test Flattened feature matrix for test data (1D vector)
 * @param rowLength Number of features in each sample
 * @param y_test Vector of target labels for test data
 * @return Mean Squared Error (MSE)
 */
double Boosting::evaluate(const std::vector<double>& X_test, int rowLength, const std::vector<double>& y_test) const {
    std::vector<double> y_pred = predict(X_test, rowLength);
    return loss_function->computeLoss(y_test, y_pred);
}

/**
 * @brief Save the Boosting model to a file
 * @param filename File to save the model
 */
void Boosting::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Save model parameters
    file << n_estimators << " "
         << learning_rate << " "
         << max_depth << " "
         << min_samples_split << " "
         << min_impurity_decrease << " "
         << initial_prediction << "\n";

    // Save each tree with a unique name
    for (size_t i = 0; i < trees.size(); ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i]->saveTree(tree_filename);
    }

    file.close();
}

/**
 * @brief Load a Boosting model from a file
 * @param filename File containing the saved model
 */
void Boosting::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Load model parameters
    file >> n_estimators
         >> learning_rate
         >> max_depth
         >> min_samples_split
         >> min_impurity_decrease
         >> initial_prediction;

    // Clear and reload trees
    trees.clear();
    trees.resize(n_estimators);

    for (int i = 0; i < n_estimators; ++i) {
        std::string tree_filename = filename + "_tree_" + std::to_string(i);
        trees[i] = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
        trees[i]->loadTree(tree_filename);
    }

    file.close();
}
