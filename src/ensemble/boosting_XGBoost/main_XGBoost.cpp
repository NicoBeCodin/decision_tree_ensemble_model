#include "boosting_XGBoost.h"
#include <iostream>

int main() {
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}
    };
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    std::unique_ptr<LossFunction> loss = std::make_unique<LeastSquaresLoss>();
    XGBoost model(10, 3, 0.1, 1.0, 0.0, std::move(loss));
    model.train(X, y);

    auto predictions = model.predict(X);
    std::cout << "Prédictions : ";
    for (const auto& pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    return 0;
}
