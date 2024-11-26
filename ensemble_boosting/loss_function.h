#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

/**
 * @brief 损失函数基类
 */
class LossFunction
{
public:
    virtual ~LossFunction() = default;

    /**
     * @brief 计算负梯度（伪残差）
     * @param y_true 真实值向量
     * @param y_pred 预测值向量
     * @return 负梯度向量
     */
    virtual std::vector<double> negativeGradient(const std::vector<double>& y_true,
                                                 const std::vector<double>& y_pred) const = 0;

    /**
     * @brief 计算损失值
     * @param y_true 真实值向量
     * @param y_pred 预测值向量
     * @return 损失值
     */
    virtual double computeLoss(const std::vector<double>& y_true,
                               const std::vector<double>& y_pred) const = 0;
};


class LeastSquaresLoss : public LossFunction
{
public:
    std::vector<double> negativeGradient(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred) const override;

    double computeLoss(const std::vector<double>& y_true,
                       const std::vector<double>& y_pred) const override;
};

#endif // LOSS_FUNCTION_H
