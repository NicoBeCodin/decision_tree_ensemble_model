#include "loss_function.h"
#include <vector>
#include <cmath>

std::vector<double> LeastSquaresLoss::negativeGradient(const std::vector<double>& y_true,
                                                       const std::vector<double>& y_pred) const {
    std::vector<double> residuals(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        residuals[i] = y_true[i] - y_pred[i];
    }
    return residuals;
}

double LeastSquaresLoss::computeLoss(const std::vector<double>& y_true,
                                     const std::vector<double>& y_pred) const {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        loss += diff * diff;
    }
    return loss / y_true.size();
}
