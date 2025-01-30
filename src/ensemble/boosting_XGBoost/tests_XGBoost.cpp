#include "../ensemble_boosting_XGBoost/boosting_XGBoost.h"
#include "../functions_tree/math_functions.h"
#include "../ensemble_boosting/loss_function.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>

//To compile : g++ -o tests_XGBoost ../ensemble_boosting_XGBoost/boosting_XGBoost.cpp ../functions_tree/decision_tree_XGBoost.cpp ../functions_tree/math_functions.cpp ../ensemble_boosting/loss_function.cpp tests_XGBoost.cpp -std=c++17



// Mock loss function for testing
class MSELoss : public LossFunction {
public:
    std::vector<double> negativeGradient(const std::vector<double>& y, const std::vector<double>& y_pred) const override {
        std::vector<double> residuals(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            residuals[i] = y[i] - y_pred[i];
        }
        return residuals;
    }

    double computeLoss(const std::vector<double>& y, const std::vector<double>& y_pred) const override {
        double loss = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            loss += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
        }
        return loss / y.size();
    }
};

// Helper function to flatten a 2D dataset into a 1D vector
std::vector<double> flattenDataset(const std::vector<std::vector<double>>& data) {
    std::vector<double> flattened;
    for (const auto& row : data) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

void testXGBoost() {
    // Sample training data
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0}
    };
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    // Flatten the dataset
    std::vector<double> X_flattened = flattenDataset(X);
    int rowLength = X[0].size();

    // Create XGBoost model
    auto loss_function = std::make_unique<MSELoss>();
    XGBoost model(20, 10,   0.01, 0.01, 0.001, std::move(loss_function));

    // Train the model
    model.train(X_flattened, rowLength, y);

    // Predict on the training set
    std::vector<double> predictions;
    for (const auto& row : X) {
        predictions.push_back(model.predict(row));
    }

    std::cout << "Predictions: ";
    for (double pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    // Evaluate the model on the training data
    double mse = model.evaluate(X_flattened, rowLength, y);
    std::cout << "MSE on training data: " << mse << std::endl;

    // Assert the MSE is within an acceptable range
    assert(mse < 0.1);

    // Save the model
    model.save("xgboost_model.txt");

    // Load the model
    XGBoost loadedModel(10, 3, 0.1, 0.01, 0.01, std::make_unique<MSELoss>());
    loadedModel.load("xgboost_model.txt");

    // Predict using the loaded model
    std::vector<double> loaded_predictions;
    for (const auto& row : X) {
        loaded_predictions.push_back(loadedModel.predict(row));
    }

    std::cout << "Loaded Model Predictions: ";
    for (double pred : loaded_predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    // Verify predictions from the loaded model match the original model
    for (size_t i = 0; i < predictions.size(); ++i) {
        assert(std::abs(predictions[i] - loaded_predictions[i]) < 1e-6);
    }

    std::cout << "All tests passed successfully!" << std::endl;
}

int main() {
    testXGBoost();
    return 0;
}
