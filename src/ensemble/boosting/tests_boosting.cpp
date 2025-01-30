#include "boosting.h"
#include "loss_function.h"  // Assuming this defines the loss functions (e.g., MSELoss)
#include <iostream>
#include <cassert>
#include <memory>


//TO compile: g++ -o tests_boosting boosting.cpp ../functions_tree/decision_tree_single.cpp ../functions_tree/math_functions.cpp loss_function.cpp tests_boosting.cpp -std=c++17


// Helper function to create a flattened dataset
std::vector<double> createFlattenedDataset(const std::vector<std::vector<double>>& data) {
    std::vector<double> flattened;
    for (const auto& row : data) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

// Custom loss function for testing
class MSELoss : public LossFunction {
public:
    std::vector<double> negativeGradient(const std::vector<double>& y, const std::vector<double>& y_pred) const override {
        std::vector<double> gradients(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            gradients[i] = y[i] - y_pred[i]; // Residuals
        }
        return gradients;
    }

    double computeLoss(const std::vector<double>& y, const std::vector<double>& y_pred) const override {
        double mse = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            mse += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
        }
        return mse / y.size();
    }
};

void testBoosting() {
    // Define dataset and labels
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
        {5.0, 6.0}
    };
    std::vector<double> labels = {1.5, 2.5, 3.5, 4.5, 5.5};

    // Flatten the dataset
    std::vector<double> flattenedData = createFlattenedDataset(data);
    int rowLength = data[0].size();

    // Create boosting model with MSE loss
    auto lossFunction = std::make_unique<MSELoss>();
    Boosting model(50, 0.05, std::move(lossFunction), 5, 2, 0.01);

    // Train the model
    model.train(flattenedData, rowLength, labels, 0); // 0 for MSE criterion

    // Make predictions
    std::vector<double> predictions = model.predict(flattenedData, rowLength);

    // Display predictions
    std::cout << "Predictions:" << std::endl;
    for (const auto& pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    // Evaluate the model
    double mse = model.evaluate(flattenedData, rowLength, labels);
    std::cout << "MSE on training data: " << mse << std::endl;

    // Check that MSE is reasonable (since it is training data, it should be low)
    assert(mse < 0.1);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    testBoosting();
    return 0;
}
