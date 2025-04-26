#include "getModelParameters.h"

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
            std::cout << "Maximum depth (default: 60): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 2): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 1e-12): ";
            std::cin >> min_impurity;
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of concurrent threads supported by the implementation: "<< availableThreads<< "\nHow many do you want to use ?\nPlease use a power of two (1,2,4,8,16 etc...)";
            std::cin>>numThreads;


            std::cout << "The criteria is : " << criteria << std::endl;
            
            parameters += " " + std::to_string(criteria) + " " + std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) +" " + 
                         std::to_string(numThreads);
            break;
        }
        case 2: {  // Bagging
            int num_trees, max_depth, min_samples;
            int criteria = -1;
            int which_loss_func = -1;
            double min_impurity;
            int numThreads = 1;
            
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
            std::cout << "Number of trees (default: 20): ";
            std::cin >> num_trees;
            std::cout << "Maximum depth (default: 60): ";
            std::cin >> max_depth;
            std::cout << "Minimum samples to split (default: 2): ";
            std::cin >> min_samples;
            std::cout << "Minimum impurity decrease (default: 1e-6): ";
            std::cin >> min_impurity;
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of concurrent threads supported by the implementation: "<< availableThreads<< "\nHow many do you want to use ? ";
            std::cin >> numThreads;
            
            parameters += " " + std::to_string(criteria) + " " + 
                         std::to_string(which_loss_func) + " " + 
                         std::to_string(num_trees) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) + " " +
                         std::to_string(numThreads);
            break;
        }
        case 3: {  // Boosting
            int n_estimators, max_depth, min_samples, numThreads;
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
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of concurrent threads supported by the implementation: "<< availableThreads<< "\nHow many do you want to use ? ";
            std::cin>> numThreads;
            
            parameters += " " + std::to_string(criteria) + " " + 
                         std::to_string(which_loss_func) + " " + 
                         std::to_string(n_estimators) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) + " " +
                         std::to_string(learning_rate) + " " +
                         std::to_string(numThreads);
            break;
        }
    }
}