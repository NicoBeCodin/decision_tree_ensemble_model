#include "boosting.h"
#include <numeric>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <thread>

/**
 * @brief Constructor to initialize the Boosting model
 * @param n_estimators Number of weak learners (decision trees)
 * @param max_depth Maximum depth for each tree
 * @param learning_rate Learning rate
 * @param loss_function Loss function to calculate gradients and errors
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
    
    // Initialiser tous les arbres au début
    trees.clear();
    trees.reserve(n_estimators);
    
    for (int i = 0; i < n_estimators; ++i) {
        trees.push_back(std::make_unique<DecisionTreeSingle>(
            max_depth, min_samples_split, min_impurity_decrease, criteria, 
            // Utiliser plusieurs threads pour chaque arbre
            #ifdef USE_OPENMP
            std::thread::hardware_concurrency(), true
            #else
            1, false
            #endif
        ));
    }
    
    // Training loop
    for (int i = 0; i < n_estimators; ++i) {
        // Calculate residuals (negative gradients)
        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

        // Train the current weak learner
        trees[i]->train(X, rowLength, residuals, criteria);

        // Update predictions en parallèle si OpenMP est disponible
        #ifdef USE_OPENMP
        #pragma omp parallel for
        for (size_t j = 0; j < n_samples; ++j) {
            std::vector<double> sample(X.begin() + j * rowLength, X.begin() + (j + 1) * rowLength);
            y_pred[j] += learning_rate * trees[i]->predict(sample);
        }
        #else
        // Version séquentielle
        for (size_t j = 0; j < n_samples; ++j) {
            std::vector<double> sample(X.begin() + j * rowLength, X.begin() + (j + 1) * rowLength);
            y_pred[j] += learning_rate * trees[i]->predict(sample);
        }
        #endif

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
double Boosting::predict(const std::vector<double>& x) const {
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
    // Vérifier que le nom de fichier est valide
    if (filename.empty()) {
        throw std::invalid_argument("Le nom de fichier ne peut pas être vide");
    }

    // Créer le répertoire parent si nécessaire
    std::string directory;
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        directory = filename.substr(0, last_slash);
        // Créer le répertoire si nécessaire (utiliser un système indépendant de la plateforme)
        #ifdef _WIN32
        std::string command = "mkdir \"" + directory + "\" 2> nul";
        #else
        std::string command = "mkdir -p \"" + directory + "\" 2>/dev/null";
        #endif
        system(command.c_str());
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Impossible d'ouvrir le fichier pour l'écriture: " + filename);
    }

    // Sauvegarder les paramètres du modèle
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
        // Utiliser un chemin relatif au fichier principal
        std::string base_filename = filename;
        size_t last_dot = base_filename.find_last_of(".");
        if (last_dot != std::string::npos) {
            base_filename = base_filename.substr(0, last_dot);
        }
        std::string tree_filename = base_filename + "_tree_" + std::to_string(i) + ".tree";
        
        try {
            trees[i]->saveTree(tree_filename);
        } catch (const std::exception& e) {
            file.close();
            throw std::runtime_error("Erreur lors de la sauvegarde de l'arbre " + 
                                    std::to_string(i) + ": " + e.what());
        }
    }

    file.close();
}

/**
 * @brief Load a Boosting model from a file
 * @param filename File containing the saved model
 */
void Boosting::load(const std::string& filename) {
    // Vérifier que le nom de fichier est valide
    if (filename.empty()) {
        throw std::invalid_argument("Le nom de fichier ne peut pas être vide");
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Impossible d'ouvrir le fichier pour la lecture: " + filename);
    }

    try {
        // Charger les paramètres du modèle
        file >> n_estimators
             >> learning_rate
             >> max_depth
             >> min_samples_split
             >> min_impurity_decrease
             >> initial_prediction
             >> Criteria
             >> whichLossFunc;
        
        // Vérifier que les paramètres sont valides
        if (n_estimators <= 0 || learning_rate <= 0 || max_depth <= 0 || 
            min_samples_split <= 0 || min_impurity_decrease < 0) {
            throw std::runtime_error("Paramètres de modèle invalides dans le fichier");
        }
        
        // Réinitialiser et recharger les arbres
        trees.clear();
        trees.resize(n_estimators);

        for (int i = 0; i < n_estimators; ++i) {
            // Utiliser un chemin relatif au fichier principal
            std::string base_filename = filename;
            size_t last_dot = base_filename.find_last_of(".");
            if (last_dot != std::string::npos) {
                base_filename = base_filename.substr(0, last_dot);
            }
            std::string tree_filename = base_filename + "_tree_" + std::to_string(i) + ".tree";
            
            trees[i] = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
            
            try {
                trees[i]->loadTree(tree_filename);
            } catch (const std::exception& e) {
                throw std::runtime_error("Erreur lors du chargement de l'arbre " + 
                                        std::to_string(i) + ": " + e.what());
            }
        }
    } catch (const std::exception& e) {
        file.close();
        throw std::runtime_error("Erreur lors du chargement du modèle: " + std::string(e.what()));
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