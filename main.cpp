#include "functions_io/functions_io.h"          // Include data input/output class
#include "functions_tree/splitting_criteria.h"  // Include splitting criteria
#include "ensemble_bagging/bagging.h"           // Include bagging class
#include "ensemble_boosting/boosting.h"         // Include boosting class
#include <iostream>
#include <string>

int main() {
    // 1. Load data
    DataIO data_io;
    auto [X, y] = data_io.readCSV("/Users/doriandrivet/test_Tree/decision_tree_ensemble_model/datasets/sample_400_rows.csv");  // Load feature matrix X and target vector y from CSV file

    // 2. Define splitting criteria
    MeanSquaredError mse;  // Use Mean Squared Error (MSE) as the splitting criterion

    // 3. Select method: Bagging or Boosting
    std::string method;
    std::cout << "Enter the ensemble method to use (bagging/boosting): ";
    std::cin >> method;

    if (method == "bagging") {
        std::cout << "Running Bagging method..." << std::endl;  // Message indicating Bagging is selected
        int num_trees = 10;        // Number of trees in the ensemble
        int max_depth = 5;         // Maximum depth of each tree
        Bagging bagging_model(num_trees, max_depth, &mse);  // Initialize bagging model

        // Train the bagging model
        bagging_model.train(X, y);

        // Evaluate the bagging model
        auto [X_test, y_test] = data_io.readCSV("/Users/doriandrivet/test_Tree/decision_tree_ensemble_model/datasets/sample_400_rows.csv");  // Load test set
        double mse_value = bagging_model.evaluate(X_test, y_test);
        std::cout << "Bagging Model Mean Squared Error (MSE): " << mse_value << std::endl;

        // Make predictions and save results
        std::vector<double> predictions;
        for (const auto& sample : X_test) {
            predictions.push_back(bagging_model.predict(sample));
        }
        data_io.writeResults(predictions, "bagging_predictions.csv");

    } else if (method == "boosting") {
        std::cout << "Running Boosting method..." << std::endl;  // Message indicating Boosting is selected
        int num_trees = 10;        // Number of trees in the boosting ensemble
        int max_depth = 5;         // Maximum depth of each tree
        Boosting boosting_model(num_trees, max_depth, &mse);  // Initialize boosting model

        // Train the boosting model
        boosting_model.train(X, y);

        // Evaluate the boosting model
        auto [X_test, y_test] = data_io.readCSV("/Users/doriandrivet/test_Tree/decision_tree_ensemble_model/datasets/sample_400_rows.csv");  // Load test set
        double mse_value = boosting_model.evaluate(X_test, y_test);
        std::cout << "Boosting Model Mean Squared Error (MSE): " << mse_value << std::endl;

        // Make predictions and save results
        std::vector<double> predictions;
        for (const auto& sample : X_test) {
            predictions.push_back(boosting_model.predict(sample));
        }
        data_io.writeResults(predictions, "boosting_predictions.csv");

    } else {
        std::cerr << "Invalid method. Please enter either 'bagging' or 'boosting'." << std::endl;
        return 1;
    }

    return 0;
}
