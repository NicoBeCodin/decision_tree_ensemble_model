#ifndef BOOSTING_H
#define BOOSTING_H

#include "../../functions/tree/decision_tree_single.h"
#include "../../functions/loss/loss_function.h"
#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

/**
 * @brief Boosting class
 * Implements gradient boosting algorithm using multiple weak learners (decision
 * trees)
 */
class Boosting {
public:
  /**
   * @brief Constructor to initialize the boosting model
   * @param n_estimators Number of weak learners (decision trees)
   * @param max_depth Maximum depth for each tree
   * @param learning_rate Learning rate for the model
   * @param criteria Splitting criterion for the trees
   * @param loss_function Loss function to minimize
   * @param numThreads Maximum number of threads
   */
  Boosting(int n_estimators, double learning_rate,
           std::unique_ptr<LossFunction> loss_function,
           int max_depth, int min_samples_split, double min_impurity_decrease, 
           int Criteria = 0, int whichLossFunc = 0, int numThreads = 1);

  /**
   * @brief Train the boosting model
   * @param X Training feature matrix
   * @param y Training target vector
   */
  void train(const std::vector<double> &X,
             int rowLength, const std::vector<double> &y, 
             const int Criteria);

  /**
   * @brief Predict a value for a given sample
   * @param x Vector representing a sample
   * @return Prediction for the sample
   */
  double predict(const double* x_ptr, int rowLength) const;

  /**
   * @brief Predict values for a set of samples
   * @param X Matrix representing multiple samples
   * @return Vector of predictions for each sample
   */
  std::vector<double> predict(const std::vector<double> &X,
                              int rowLength) const;

  /**
   * @brief Evaluate the model on a test set
   * @param X_test Test feature matrix
   * @param y_test Test target vector
   * @return Mean error of predictions on the test set
   */
  double evaluate(const std::vector<double> &X_test, int rowLength,
                  const std::vector<double> &y_test) const;

  // New method to access the estimators
  const std::vector<std::unique_ptr<DecisionTreeSingle>> & getEstimators() const { return trees; }

  // Serialization methods
  void save(const std::string &filename) const;
  void load(const std::string &filename);
  double getInitialPrediction() const { return initial_prediction; }

  std::map<std::string, std::string> getTrainingParameters() const;
  std::string getTrainingParametersString() const;

  

private:
  int n_estimators;
  int max_depth;
  int min_samples_split;
  double min_impurity_decrease;
  double learning_rate;
  int Criteria;
  int whichLossFunc;
  std::unique_ptr<LossFunction> loss_function;

  int numThreads;

  std::vector<std::unique_ptr<DecisionTreeSingle>> trees; // Collection of weak learners
  double initial_prediction;

  /**
   * @brief Initialize the initial prediction based on the target vector
   * @param y Target vector
   */
  void initializePrediction(const std::vector<double> &y);

  friend class FeatureImportance;
};

#endif // BOOSTING_H