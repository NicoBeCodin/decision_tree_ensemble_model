#include "getModelParameters.h"

void getModelParameters(int model_choice, std::string& parameters) {
    bool input = false;
    bool load_existing = false;
    std::cout << "Would you like to load an existing model? (1 = Yes, 0 = No): ";
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
            case 4: {  // LightGBM
                std::string model_filename;
                std::cout << "Enter the filename of the LightGBM model to load: ";
                std::cin >> model_filename;
                std::string path = "../saved_models/lightgbm_models/" + model_filename;
                parameters += " " + path;
                return;
            }
            case 5: {  // Advanced GBDT
                std::string model_filename;
                std::cout << "Enter the filename of the Advanced GBDT model to load: ";
                std::cin >> model_filename;
                std::string path = "../saved_models/adv_gbdt_models/" + model_filename;
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
            int numThreads = 1;
            bool useOMP = false;
            double min_impurity;
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
            std::cout << "Use OMP (1 = Yes, 0 = No (default): ";
            std::cin >> useOMP;
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of concurrent threads supported by the implementation: "<< availableThreads << "\nHow many do you want to use ?\nPlease use a power of two (1,2,4,8,16 etc...)";
            std::cin>>numThreads;


            std::cout << "The criteria is : " << criteria << std::endl;
            
            parameters += " " + std::to_string(criteria) + " " + std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) + " " +
                         std::to_string(useOMP) + " " +
                         std::to_string(numThreads);
            break;
        }
        case 2: {  // Bagging
            int num_trees, max_depth, min_samples;
            int criteria = -1;
            int which_loss_func = -1;
            double min_impurity;
            int numThreads = 1;
            bool useOMP = false;
            
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
            std::cout << "Use OMP (1 = Yes, 0 = No (default): ";
            std::cin >> useOMP;
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of concurrent threads supported by the implementation: "<< availableThreads<< "\nHow many do you want to use ? ";
            std::cin >> numThreads;
            
            parameters += " " + std::to_string(criteria) + " " + 
                         std::to_string(which_loss_func) + " " + 
                         std::to_string(num_trees) + " " +
                         std::to_string(max_depth) + " " + 
                         std::to_string(min_samples) + " " + 
                         std::to_string(min_impurity) + " " +
                         std::to_string(useOMP) + " " +
                         std::to_string(numThreads);
            break;
        }
        case 3: {  // Boosting
            int n_estimators, max_depth, min_samples, numThreads;
            int criteria = -1;
            int which_loss_func = -1;
            bool useOMP = false;
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
            std::cout << "Use OMP (1 = Yes, 0 = No (default): ";
            std::cin >> useOMP;
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
                         std::to_string(useOMP) + " " +
                         std::to_string(numThreads);
            break;
        }
        case 4: {  // LightGBM
            int n_estimators, max_depth, num_leaves;
            double learning_rate, subsample, colsample_bytree;
            
            std::cout << "\nLightGBM Parameters:\n";
            std::cout << "Number of estimators (default: 100): ";
            std::cin >> n_estimators;
            std::cout << "Learning rate (default: 0.1): ";
            std::cin >> learning_rate;
            std::cout << "Maximum depth (-1 for no limit, default: -1): ";
            std::cin >> max_depth;
            std::cout << "Number of leaves (default: 31): ";
            std::cin >> num_leaves;
            std::cout << "Subsample ratio (default: 1.0): ";
            std::cin >> subsample;
            std::cout << "Column sample by tree ratio (default: 1.0): ";
            std::cin >> colsample_bytree;
            
            parameters += " " + std::to_string(n_estimators) + " " + 
                         std::to_string(learning_rate) + " " + 
                         std::to_string(max_depth) + " " +
                         std::to_string(num_leaves) + " " + 
                         std::to_string(subsample) + " " + 
                         std::to_string(colsample_bytree);
            break;
        }
        case 5: {  // Advanced GBDT
            int n_estimators, max_depth, min_data_leaf, num_bins, num_threads;
            int binning_method, use_dart;
            double learning_rate, dropout_rate, skip_drop_rate, l2_reg, feature_sample_ratio;
            int early_stopping_rounds;
            
            std::cout << "\nAdvanced GBDT Parameters:\n";
            std::cout << "Number of estimators (default: 200): ";
            std::cin >> n_estimators;
            std::cout << "Learning rate (default: 0.01): ";
            std::cin >> learning_rate;
            std::cout << "Maximum depth (default: 50): ";
            std::cin >> max_depth;
            std::cout << "Minimum data in leaf (default: 1): ";
            std::cin >> min_data_leaf;
            std::cout << "Number of bins (default: 1024): ";
            std::cin >> num_bins;
            std::cout << "Use DART (1 = Yes, 0 = No, default: 1): ";
            std::cin >> use_dart;
            
            if (use_dart) {
                std::cout << "Dropout rate (default: 0.5): ";
                std::cin >> dropout_rate;
                std::cout << "Skip dropout rate (default: 0.3): ";
                std::cin >> skip_drop_rate;
            } else {
                dropout_rate = 0.0;
                skip_drop_rate = 0.0;
            }
            
            std::cout << "Binning method (0 = Quantile, 1 = Frequency, default: 1): ";
            std::cin >> binning_method;
            std::cout << "L2 regularization (default: 1.0): ";
            std::cin >> l2_reg;
            std::cout << "Feature sampling ratio (0.0-1.0, default: 1.0): ";
            std::cin >> feature_sample_ratio;
            std::cout << "Early stopping rounds (0 to disable, default: 0): ";
            std::cin >> early_stopping_rounds;
            
            int availableThreads = std::thread::hardware_concurrency();
            std::cout << "Number of threads (detected: " << availableThreads << ", default: 8): ";
            std::cin >> num_threads;
            
            parameters += " " + std::to_string(n_estimators) + " " + 
                         std::to_string(learning_rate) + " " + 
                         std::to_string(max_depth) + " " +
                         std::to_string(min_data_leaf) + " " + 
                         std::to_string(num_bins) + " " + 
                         std::to_string(use_dart) + " " +
                         std::to_string(dropout_rate) + " " +
                         std::to_string(skip_drop_rate) + " " +
                         std::to_string(num_threads) + " " +
                         std::to_string(binning_method) + " " +
                         std::to_string(l2_reg) + " " +
                         std::to_string(feature_sample_ratio) + " " +
                         std::to_string(early_stopping_rounds);
            break;
        }
    }
}