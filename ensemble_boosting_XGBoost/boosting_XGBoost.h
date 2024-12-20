#ifndef BOOSTING_XGBOOST_H
#define BOOSTING_XGBOOST_H

#include "../functions_tree/decision_tree_XGBoost.h"
#include "../ensemble_boosting/loss_function.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <map>

/**
 * @brief Main class implementing XGBoost.
 */
class XGBoost {
private:
    int n_estimators;
    int max_depth;
    double learning_rate;
    double lambda;  // L2 regularization
    double alpha;   // L1 regularization
    std::unique_ptr<LossFunction> loss_function;
    std::vector<std::unique_ptr<DecisionTreeXGBoost>> trees;
    double initial_prediction;
    int whichLossFunc;

    /**
     * @brief Initialize the initial prediction with the mean of the y values.
     * @param y Target labels vector
     */
    void initializePrediction(const std::vector<double>& y);

public:
    /**
     * @brief Constructor to initialize the XGBoost model for boosting
     * @param n_estimators Number of weak learners (decision trees)
     * @param max_depth Maximum depth for each tree
     * @param learning_rate Learning rate
     * @param lambda L2 regularization parameter
     * @param alpha L1 regularization parameter
     * @param loss_function Loss function (to compute the gradient and loss)
     */
    XGBoost(int n_estimators, int max_depth, double learning_rate, double lambda, double alpha, std::unique_ptr<LossFunction> loss_function, int whichLossFunc);
    
    /**
     * @brief Train the Boosting model
     * @param X Feature matrix
     * @param y Target labels vector
     */
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    
    /**
     * @brief Predict for a single sample
     * @param x Feature vector of a sample
     * @return Prediction for the sample
     */
    double predict(const std::vector<double>& x) const;

    /**
     * @brief Predict for multiple samples
     * @param X Feature matrix
     * @return Vector of predictions for each sample
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    
    /**
     * @brief Evaluate the model performance on a test set
     * @param X_test Test feature matrix
     * @param y_test Test target labels vector
     * @return Mean Squared Error (MSE)
     */
    double evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const;

    /**
     * @brief Compute feature importance
     * @param feature_names Feature names (optional)
     * @return Map of features with their relative importance
     */
    std::map<std::string, double> featureImportance(const std::vector<std::string>& feature_names = {}) const;
    
    /** 
     * @brief Destructor of XGBoost
    */
    ~XGBoost() = default;

    // Serialization methods
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    double getInitialPrediction() const { return initial_prediction; }

    std::map<std::string, std::string> getTrainingParameters() const;
    std::string getTrainingParametersString() const;
};

#endif // BOOSTING_XGBOOST_H
