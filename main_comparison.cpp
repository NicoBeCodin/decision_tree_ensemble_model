#include "model_comparison/model_comparison.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>

void getModelParameters(int model_choice, std::string& parameters) {
    std::string input;
    std::cout << "\nDo you want to customize parameters? (y/n): ";
    std::cin >> input;
    
    if (input != "y" && input != "Y") {
        return;  // Use default parameters
    }

    parameters += " -p";  // Flag to indicate custom parameters

    switch(model_choice) {
        case 1: {  // Single Tree
            int max_depth, min_samples;
            double min_impurity;
            
            std::cout << "\nDecision Tree Parameters:\n";
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
            double min_impurity;
            
            std::cout << "\nBagging Parameters:\n";
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
            double min_impurity, learning_rate;
            
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
            std::string command = "echo \"" + parameters + "\" | ./MainEnsemble";
            system(command.c_str());
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