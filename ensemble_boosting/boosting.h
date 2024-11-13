#ifndef BOOSTING_H
#define BOOSTING_H

#include "../functions_tree/regression_tree.h"
#include "loss_function.h"
#include <vector>
#include <memory>

class Boosting {
public:
    Boosting(int n_estimators, int max_depth, double learning_rate,
             SplittingCriteria* criteria, std::unique_ptr<LossFunction> loss_function);

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& y);

    double predict(const std::vector<double>& x) const;

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    double evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const;


private:
    int n_estimators;   // 弱学习器数量
    int max_depth;      // 每个弱学习器的最大深度
    double learning_rate; // 学习率
    SplittingCriteria* criteria;  // 分裂标准，用于创建每个树的分裂规则
    std::unique_ptr<LossFunction> loss_function; // 损失函数

    std::vector<std::unique_ptr<RegressionTree>> estimators; // 弱学习器集合
    double initial_prediction; // 初始预测值（常数模型）

    void initializePrediction(const std::vector<double>& y);

};

#endif // BOOSTING_H
