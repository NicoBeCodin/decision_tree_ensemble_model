#include "model_comparison/model_comparison.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <chrono>
#include <limits>

void getModelParameters(int model_choice, std::string& parameters) {
    bool input = false;
    bool load_existing = false;
    std::cout << "Would you like to load an existing tree model? (1 = Yes (for the moment, no use), 0 = No): ";
    std::cin >> load_existing;
    // Vider le buffer de l'entrée pour éviter les caractères résiduels
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if (load_existing) {
        parameters += " -l";
        switch (model_choice) {
            case 1: { // single tree
                std::string model_filename;
                std::cout << "Enter the filename of the tree model to load: ";
                std::cin >> model_filename;

                std::string path = "../saved_models/tree_models/" + model_filename;

                parameters += " " + path;

                return;
            }
            case 2: {  // Bagging
                std::string model_filename;
                std::cout << "Enter the filename of the bagging model to load: ";
                std::cin >> model_filename;

                std::string path = "../saved_models/bagging_models/" + model_filename;

                parameters += " " + path;

                return;
            }
            case 3: {  // Boosting
                std::string model_filename;
                std::cout << "Enter the filename of the boosting model to load: ";
                std::cin >> model_filename;

                std::string path = "../saved_models/boosting_models/" + model_filename;

                parameters += " " + path;

                return;
            }
            case 4: {  // XGBoost
                std::string model_filename;
                std::cout << "Enter the filename of the xgboost model to load: ";
                std::cin >> model_filename;

                std::string path = "../saved_models/xgboost_models/" + model_filename;

                parameters += " " + path;

                return;
            }
        }
    }
    std::cout << "\nDo you want to customize parameters? (1 = Yes, 0 = No): ";
    std::cin >> input;
    // Vider le buffer de l'entrée pour éviter les caractères résiduels
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    if (!input) {
        return;  // Use default parameters
    }

    parameters += " -p";  // Flag to indicate custom parameters

    switch(model_choice) {
        case 1: {  // Single Tree
            int max_depth, min_samples;
            int criteria = -1;
            double min_impurity;
            
            std::cout << "\nDecision Tree Parameters:\n";
            // Boucle jusqu'à ce que l'utilisateur entre 0 ou 1 pour criteria
            while (criteria != 0 && criteria != 1) {
                std::cout << "Which method do you want as a splitting criteria: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> criteria;

                // Vérification de l'entrée de l'utilisateur
                if (std::cin.fail() || (criteria != 0 && criteria != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Nettoyer le flux d'entrée
                    std::cin.clear(); // Réinitialise l'état du flux
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore tout ce qui se trouve dans le flux d'entrée
                }
            }
            std::cout << "Maximum depth (default: 5): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 2): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 0.0): ";
            std::cin >> min_impurity;
            
            parameters += " " + std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity);
            break;
        }
        case 2: {  // Bagging
            int n_estimators, max_depth, min_samples;
            int criteria = -1;
            int which_loss_func = -1;
            double min_impurity;
            
            std::cout << "\nBagging Parameters:\n";
            // Boucle jusqu'à ce que l'utilisateur entre 0 ou 1 pour criteria
            while (criteria != 0 && criteria != 1) {
                std::cout << "Which method do you want as a splitting criteria: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> criteria;

                // Vérification de l'entrée de l'utilisateur
                if (std::cin.fail() || (criteria != 0 && criteria != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Nettoyer le flux d'entrée
                    std::cin.clear(); // Réinitialise l'état du flux
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore tout ce qui se trouve dans le flux d'entrée
                }
            }
            // Boucle jusqu'à ce que l'utilisateur entre 0 ou 1 pour which_loss_func
            while (which_loss_func != 0 && which_loss_func != 1) {
                std::cout << "Which method do you want as a comparing trees: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> which_loss_func;

                // Vérification de l'entrée de l'utilisateur
                if (std::cin.fail() || (which_loss_func != 0 && which_loss_func != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Nettoyer le flux d'entrée
                    std::cin.clear(); // Réinitialise l'état du flux
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore tout ce qui se trouve dans le flux d'entrée
                }
            }
            std::cout << "Number of estimators (default: 10): ";
            std::cin >> n_estimators;
            std::cout << "Maximum depth (default: 5): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 2): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 0.0): ";
            std::cin >> min_impurity;
            
            parameters += " " + std::to_string(n_estimators) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity);
            break;
        }
        case 3: {  // Boosting
            int n_estimators, max_depth, min_samples;
            int criteria = -1;
            int which_loss_func = -1;
            double min_impurity, learning_rate;
            
            // Boucle jusqu'à ce que l'utilisateur entre 0 ou 1 pour criteria
            while (criteria != 0 && criteria != 1) {
                std::cout << "Which method do you want as a splitting criteria: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> criteria;

                // Vérification de l'entrée de l'utilisateur
                if (std::cin.fail() || (criteria != 0 && criteria != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Nettoyer le flux d'entrée
                    std::cin.clear(); // Réinitialise l'état du flux
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore tout ce qui se trouve dans le flux d'entrée
                }
            }
            // Boucle jusqu'à ce que l'utilisateur entre 0 ou 1 pour which_loss_func
            while (which_loss_func != 0 && which_loss_func != 1) {
                std::cout << "Which method do you want as a comparing trees: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> which_loss_func;

                // Vérification de l'entrée de l'utilisateur
                if (std::cin.fail() || (which_loss_func != 0 && which_loss_func != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Nettoyer le flux d'entrée
                    std::cin.clear(); // Réinitialise l'état du flux
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore tout ce qui se trouve dans le flux d'entrée
                }
            }
            std::cout << "\nBoosting Parameters:\n";
            std::cout << "Number of estimators (default: 75): ";
            std::cin >> n_estimators;
            std::cout << "Maximum depth (default: 15): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 3): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 1e-5): ";
            std::cin >> min_impurity;
            std::cout << "Learning rate (default: 0.07): ";
            std::cin >> learning_rate;
            
            parameters += " " + std::to_string(n_estimators) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) + " " +
                         std::to_string(learning_rate);
            break;
        }
        case 4: {  // XGBoost
            int n_estimators, max_depth;
            double learning_rate, lambda, gamma;
            
            std::cout << "\nXGBoost Parameters:\n";
            std::cout << "Number of estimators (default: 10): ";
            std::cin >> n_estimators;
            std::cout << "Maximum depth (default: 5): ";
            std::cin >> max_depth;
            std::cout << "Learning rate (default: 0.1): ";
            std::cin >> learning_rate;
            std::cout << "Lambda - L2 regularization (default: 1.0): ";
            std::cin >> lambda;
            std::cout << "Gamma - complexity regularization (default: 0.0): ";
            std::cin >> gamma;
            
            parameters += " " + std::to_string(n_estimators) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(learning_rate) + " " +
                         std::to_string(lambda) + " " +
                         std::to_string(gamma);
            break;
        }
    }
}

int main() {
    std::cout << "Decision Tree Models Comparison Program\n\n";

    int choice;
    std::cout << "Choose an option:\n";
    std::cout << "1. Run individual model\n";
    std::cout << "2. Run all tests\n";
    std::cout << "3. View models comparison\n";
    std::cin >> choice;

    switch (choice) {
        case 1: {
            std::cout << "\nChoose model to use:\n";
            std::cout << "1. Single Decision Tree\n";
            std::cout << "2. Bagging\n";
            std::cout << "3. Boosting\n";
            std::cout << "4. XGBoost\n";
            
            int model_choice;
            std::cin >> model_choice;

            std::string parameters = std::to_string(model_choice);
            getModelParameters(model_choice, parameters);

            // Build command with parameters
            std::string command = "./MainEnsemble " + parameters;
            system((command + " 2>&1").c_str());
            break;
        }
        case 2: {
            std::cout << "\nRunning all tests...\n\n";
            
            std::cout << "=== Math Functions Tests ===\n";
            system("./math_functions_test");
            
            std::cout << "\n=== Decision Tree Tests ===\n";
            system("./decision_tree_test");
            
            std::cout << "\n=== Bagging Tests ===\n";
            system("./bagging_test");
            
            std::cout << "\n=== Boosting Tests ===\n";
            system("./boosting_test");
            
            std::cout << "\n=== XGBoost Tests ===\n";
            system("./xgboost_test");
            
            std::cout << "\n=== Cross Validation Tests ===\n";
            system("./cross_validation_test");
            
            std::cout << "\nAll tests completed.\n";
            break;
        }
        case 3: {
            std::cout << "\nDisplaying previous results...\n";
            std::ifstream file("../results/all_models_comparison.md");
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    std::cout << line << '\n';
                }
                file.close();
            } else {
                std::cout << "No previous results found. Please run tests first.\n";
            }
            break;
        }
        default:
            std::cout << "Invalid option\n";
            return 1;
    }

    return 0;
} 