#include "boosting_XGBoost.h"
#include <iostream>

int main() {
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}
    };
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    auto loss_function = std::make_unique<LossFunction>();
    XGBoost model(10, 3, 0.1, 1.0, 0.0, std::move(loss_function));
    model.train(X, y);

    auto predictions = model.predict(X);
    std::cout << "Prédictions : ";
    for (const auto& pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    return 0;
}
