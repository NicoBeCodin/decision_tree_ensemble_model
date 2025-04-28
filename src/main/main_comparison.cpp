#include "../pipeline/getModelParameters.h"

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
