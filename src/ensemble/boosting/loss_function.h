#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

/**
 * @brief Base class for loss functions
 */
class LossFunction
{
public:
    virtual ~LossFunction() = default;

    /**
     * @brief Compute negative gradient (pseudo-residuals)
     * @param y_true Vector of true values
     * @param y_pred Vector of predicted values
     * @return Vector of negative gradients
     */
    virtual std::vector<double> negativeGradient(const std::vector<double>& y_true,
                                                 const std::vector<double>& y_pred) const = 0;

    /**
     * @brief Compute loss value
     * @param y_true Vector of true values
     * @param y_pred Vector of predicted values
     * @return Loss value
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

class MeanAbsoluteLoss: public LossFunction{
    public:
    std::vector<double> negativeGradient(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred) const override;

    double computeLoss(const std::vector<double>& y_true,
                       const std::vector<double>& y_pred) const override;

};

#endif // LOSS_FUNCTION_H
