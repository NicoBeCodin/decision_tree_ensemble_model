#include "boosting.h"
#include <numeric>
#include <iostream>

Boosting::Boosting(int n_estimators, int max_depth, double learning_rate, SplittingCriteria* criteria,
                   std::unique_ptr<LossFunction> loss_function)
    : n_estimators(n_estimators),
      max_depth(max_depth),
      learning_rate(learning_rate),
      criteria(criteria),
      loss_function(std::move(loss_function)),
      initial_prediction(0.0) {}

void Boosting::initializePrediction(const std::vector<double>& y) {
    // 初始化预测值为 y 的均值
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

void Boosting::train(const std::vector<std::vector<double>>& X,
                     const std::vector<double>& y) {
    size_t n_samples = y.size();
    initializePrediction(y);
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (int i = 0; i < n_estimators; ++i) {
        // 计算负梯度（伪残差）
        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

        // 训练弱学习器（回归树）拟合残差
        auto tree = std::make_unique<RegressionTree>(max_depth, criteria);
        tree->train(X, residuals);

        // 更新预测值
        for (size_t j = 0; j < n_samples; ++j) {
            y_pred[j] += learning_rate * tree->predict(X[j]);
        }

        // 保存弱学习器
        estimators.push_back(std::move(tree));

        // 可选：输出当前损失值
        double loss = loss_function->computeLoss(y, y_pred);
        std::cout << "Estimator " << i + 1 << ", Loss: " << loss << std::endl;
    }
}

double Boosting::predict(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& tree : estimators) {
        y_pred += learning_rate * tree->predict(x);
    }
    return y_pred;
}

std::vector<double> Boosting::predict(const std::vector<std::vector<double>>& X) const {
    size_t n_samples = X.size();
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (const auto& tree : estimators) {
        for (size_t i = 0; i < n_samples; ++i) {
            y_pred[i] += learning_rate * tree->predict(X[i]);
        }
    }
    return y_pred;
}
double Boosting::evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const {
    double total_error = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        double prediction = predict(X_test[i]);
        total_error += std::pow(prediction - y_test[i], 2);
    }
    return total_error / X_test.size();  // 返回均方误差 (MSE)
}
