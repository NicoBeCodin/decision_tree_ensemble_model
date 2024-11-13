// loss_function.h

#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

class LossFunction {
public:
    virtual ~LossFunction() = default;

    // 计算负梯度（伪残差）
    virtual std::vector<double> negativeGradient(const std::vector<double>& y_true,
                                                 const std::vector<double>& y_pred) const = 0;

    // 计算损失值
    virtual double computeLoss(const std::vector<double>& y_true,
                               const std::vector<double>& y_pred) const = 0;
};

class LeastSquaresLoss : public LossFunction {
public:
    std::vector<double> negativeGradient(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred) const override;

    double computeLoss(const std::vector<double>& y_true,
                       const std::vector<double>& y_pred) const override;
};

#endif // LOSS_FUNCTION_H
