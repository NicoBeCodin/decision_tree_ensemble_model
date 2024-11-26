#ifndef XGBOOST_H
#define XGBOOST_H

#include "../functions_tree/decision_tree_single.h"
#include "../ensemble_boosting/loss_function.h"
#include <vector>
#include <memory>

/**
 * @brief Classe représentant un arbre de régression.
 */
class RegressionTree {
private:
    double constant_prediction;

public:
    RegressionTree();
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& residuals);
    double predict(const std::vector<double>& x) const;
};

/**
 * @brief Classe principale implémentant XGBoost.
 */
class XGBoost {
private:
    int n_estimators;
    int max_depth;
    double learning_rate;
    double lambda;
    double alpha;
    double initial_prediction;
    std::vector<std::unique_ptr<RegressionTree>> estimators;
    std::unique_ptr<LossFunction> loss_function;

    void initializePrediction(const std::vector<double>& y);

public:
    XGBoost(int n_estimators, int max_depth, double learning_rate, double lambda, double alpha, std::unique_ptr<LossFunction> loss_function);
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    ~XGBoost() = default;
};

#endif // XGBOOST_H
