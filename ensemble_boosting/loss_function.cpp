#include "loss_function.h"
#include "../functions_tree/math_functions.h"
#include <vector>
#include <cmath>

std::vector<double> LeastSquaresLoss::negativeGradient(const std::vector<double>& y_true,
                                                       const std::vector<double>& y_pred) const {
        return Math::negativeGradient(y_true, y_pred);
}

double LeastSquaresLoss::computeLoss(const std::vector<double>& y_true,
                                     const std::vector<double>& y_pred) const {
        return Math::computeLoss(y_true, y_pred);
}
