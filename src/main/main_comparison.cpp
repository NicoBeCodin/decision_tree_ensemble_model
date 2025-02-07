#include "../model_comparison/model_comparison.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <chrono>
#include <limits>
#include <thread>

void getModelParameters(int model_choice, std::string& parameters) {
    bool input = false;
    bool load_existing = false;
    std::cout << "Would you like to load an existing tree model? (1 = Yes (currently unused), 0 = No): ";
    std::cin >> load_existing;
    // Clear the input buffer to avoid residual characters
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
    // Clear the input buffer to avoid residual characters
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
            int numThreads = 1;
            
            std::cout << "\nDecision Tree Parameters:\n";
            // Loop until the user enters 0 or 1 for criteria
            while (criteria != 0 && criteria != 1) {
                std::cout << "Which method do you want as a splitting criterion: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> criteria;

                // Check user input
                if (std::cin.fail() || (criteria != 0 && criteria != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Clean input stream
                    std::cin.clear(); // Reset the input stream state
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore all remaining characters in the input stream
                }
            }
            std::cout << "Maximum depth (default: 5): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 2): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 0.0): ";
            std::cin >> min_impurity;
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of concurrent threads supported by the implementation: "<< availableThreads<< "\nHow many do you want to use ?\nPlease use a power of two (1,2,4,8,16 etc...)";
            std::cin>>numThreads;


            std::cout << "The criteria is : " << criteria << std::endl;
            
            parameters += " " + std::to_string(criteria) + " " + std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) +" " + std::to_string(numThreads);
            break;
        }
        case 2: {  // Bagging
            int num_trees, max_depth, min_samples;
            int criteria = -1;
            int which_loss_func = -1;
            double min_impurity;
            
            std::cout << "\nBagging Parameters:\n";
            // Loop until the user enters 0 or 1 for criteria
            while (criteria != 0 && criteria != 1) {
                std::cout << "Which method do you want as a splitting criterion: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> criteria;

                // Check user input
                if (std::cin.fail() || (criteria != 0 && criteria != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Clean input stream
                    std::cin.clear(); // Reset the input stream state
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore all remaining characters in the input stream
                }
            }
            // Loop until the user enters 0 or 1 for which_loss_func
            while (which_loss_func != 0 && which_loss_func != 1) {
                std::cout << "Which method do you want as a comparing trees: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> which_loss_func;

                // Check user input
                if (std::cin.fail() || (which_loss_func != 0 && which_loss_func != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Clean input stream
                    std::cin.clear(); // Reset the input stream state
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore all remaining characters in the input stream
                }
            }
            std::cout << "Number of trees (default: 10): ";
            std::cin >> num_trees;
            std::cout << "Maximum depth (default: 5): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 2): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 0.0): ";
            std::cin >> min_impurity;
            
            parameters += " " + std::to_string(criteria) + " " + 
                         std::to_string(which_loss_func) + " " + 
                         std::to_string(num_trees) + " " +
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

            std::cout << "\nBoosting Parameters:\n";
            // Loop until the user enters 0 or 1 for criteria
            while (criteria != 0 && criteria != 1) {
                std::cout << "Which method do you want as a splitting criterion: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> criteria;

                // Check user input
                if (std::cin.fail() || (criteria != 0 && criteria != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Clean input stream
                    std::cin.clear(); // Reset the input stream state
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore all remaining characters in the input stream
                }
            }
            // Loop until the user enters 0 or 1 for which_loss_func
            while (which_loss_func != 0 && which_loss_func != 1) {
                std::cout << "Which method do you want as a comparing trees: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> which_loss_func;

                // Check user input
                if (std::cin.fail() || (which_loss_func != 0 && which_loss_func != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Clean input stream
                    std::cin.clear(); // Reset the input stream state
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore all remaining characters in the input stream
                }
            }
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
            
            parameters += " " + std::to_string(criteria) + " " + 
                         std::to_string(which_loss_func) + " " + 
                         std::to_string(n_estimators) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) + " " +
                         std::to_string(learning_rate);
            break;
        }
        case 4: {  // XGBoost
            int n_estimators, max_depth, min_samples, which_loss_func;
            double learning_rate, lambda, gamma;

            std::cout << "\nXGBoost Parameters:\n";
            // Loop until the user enters 0 or 1 for which_loss_func
            while (which_loss_func != 0 && which_loss_func != 1) {
                std::cout << "Which method do you want as a comparing trees: MSE (0) or MAE (1) ?" << std::endl;
                std::cin >> which_loss_func;

                // Check user input
                if (std::cin.fail() || (which_loss_func != 0 && which_loss_func != 1)) {
                    std::cout << "Invalid input. Please enter 0 (for MSE) or 1 (for MAE)." << std::endl;

                    // Clean input stream
                    std::cin.clear(); // Reset the input stream state
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore all remaining characters in the input stream
                }
            }
            std::cout << "Number of estimators (default: 75): ";
            std::cin >> n_estimators;
            std::cout << "Maximum depth (default: 15): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 3): ";
            std::cin >> min_samples;
            std::cout << "Learning rate (default: 0.07): ";
            std::cin >> learning_rate;
            std::cout << "Lambda - L2 regularization (default: 1.0): ";
            std::cin >> lambda;
            std::cout << "Gamma - complexity regularization (default: 0.05): ";
            std::cin >> gamma;
            
            parameters += " " + std::to_string(which_loss_func) + " " + 
                         std::to_string(n_estimators) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " +
                         std::to_string(learning_rate) + " " +
                         std::to_string(lambda) + " " +
                         std::to_string(gamma);
            break;
        }
    }
}

int main() {
    std::cout << "Decision Tree Models Comparing Program\n\n";

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
            std::cout << command << std::endl;
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
