#include "functions_io/functions_io.h"       // Include data input/output class
#include "functions_tree/regression_tree.h"  // Include regression tree class
#include "functions_tree/splitting_criteria.h" // Include splitting criteria
#include <iostream>

int main() {
    // 1. Load data
    DataIO data_io;
    auto [X, y] = data_io.readCSV("/home/yifan/桌面/31_10_ppn/decision_tree_ensemble_model/datasets/sample_400_rows.csv");  // Load feature matrix X and target vector y from CSV file

    // 2. Define splitting criteria
    MeanSquaredError mse;  // Use Mean Squared Error (MSE) as the splitting criterion

    // 3. Create regression tree instance
    int maxDepth = 5;  // Maximum depth of the tree; can be adjusted as needed
    RegressionTree reg_tree(maxDepth, &mse);

    // 4. Train model
    reg_tree.train(X, y);  // Train the regression tree with the loaded data

    // 5. Model evaluation (assuming test set X_test and y_test are available in the data)
    // In practice, the test set is usually separate from the training set.
    auto [X_test, y_test] = data_io.readCSV("/home/yifan/桌面/31_10_ppn/decision_tree_ensemble_model/datasets/sample_100_rows.csv");  // Load test set data

    double mse_value = reg_tree.evaluate(X_test, y_test);  // Evaluate model performance with the test set
    std::cout << "Model Mean Squared Error (MSE): " << mse_value << std::endl;

    // 6. Model prediction
    std::vector<double> predictions;
    for (const auto& sample : X_test) {
        predictions.push_back(reg_tree.predict(sample));  // Make predictions on test set samples
    }

    // 7. Save prediction results
    data_io.writeResults(predictions, "predictions.csv");  // Save prediction results to a file

    return 0;
}
