// loss_function.h

#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>
#include "math_functions.h"

class LossFunction
{
public:
    virtual ~LossFunction() = default;

    // 计算负梯度（伪残差）
    // Calculate negative gradient (pseudo-residual)
    virtual std::vector<double> negativeGradient(const std::vector<double> &y_true,
                                                 const std::vector<double> &y_pred) const = 0;

    // 计算损失值
    // Calculate loss value
    virtual double computeLoss(const std::vector<double> &y_true,
                               const std::vector<double> &y_pred) const = 0;
};
// Calls math_functions.h
class LeastSquaresLoss : public LossFunction, public Math
{
public:
    std::vector<double> negativeGradient(const std::vector<double> &y_true,
                                         const std::vector<double> &y_pred) const override
    {
        return Math::negativeGradient(y_true, y_pred);
    }

    double computeLoss(const std::vector<double> &y_true,
                       const std::vector<double> &y_pred) const override
    {
        return Math::computeLoss(y_true, y_pred);
    }
};

#endif // LOSS_FUNCTION_H
