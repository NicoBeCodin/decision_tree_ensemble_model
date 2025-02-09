#ifndef UTILITY
#define UTILITY

#include "../ensemble/boosting_XGBoost/boosting_XGBoost.h"
#include "../functions/feature/feature_importance.h"
#include "../functions/tree/decision_tree_single.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

// Simple header file for all the utility functions to go

int adjustNumThreads(int numThreads);

// input function to set parameters with defaults
template <typename T>
T getInputWithDefault(const std::string &prompt, T defaultValue);

void displayFeatureImportance(
    const std::vector<FeatureImportance::FeatureScore> &scores);

struct ProgramOptions {
  int choice = 0;
  bool use_custom_params = false;
  bool load_request = false;
  std::string path_model_filename;
  std::vector<std::string> params;
};

ProgramOptions parseCommandLineArguments(int argc, char *argv[]);

void createDirectory(const std::string &path);

template <typename ModelType> void saveModel(ModelType &model) {
  std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
  int save_model;
  std::cin >> save_model;

  if (save_model) {
    std::cout << "Enter the filename to save the model: ";
    std::string filename;
    std::cin >> filename;
    std::string path = "../saved_models/" + filename;
    model.save(path);
    std::cout << "Model saved successfully as " << filename << "\n";
  }
}

inline void saveModel(DecisionTreeSingle &model) {
  std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
  int save_model;
  std::cin >> save_model;

  if (save_model) {
    std::cout << "Enter the filename to save the model: ";
    std::string filename;
    std::cin >> filename;
    std::string path = "../saved_models/" + filename;
    model.saveTree(path);
    std::cout << "Model saved successfully as " << filename << "\n";
  }
}

inline void saveModel(XGBoost &model) {
  std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
  int save_model;
  std::cin >> save_model;

  if (save_model) {
    std::cout << "Enter the filename to save the model: ";
    std::string filename;
    std::cin >> filename;
    std::string path = "../saved_models/" + filename;
    model.save(path);
    std::cout << "Model saved successfully as " << filename << "\n";
  }
}

template <typename ModelType>
void trainAndEvaluateModel(ModelType &model, const std::vector<double> &X_train,
                           int rowLength, const std::vector<double> &y_train,
                           const std::vector<double> &X_test,
                           const std::vector<double> &y_test, int criteria,
                           double &score, double &train_duration_count,
                           double &eval_duration_count, std::string loss_func) {
  std::cout << "Training process started, please wait...\n";

  auto train_start = std::chrono::high_resolution_clock::now();
  if (criteria != -1) {
    model.train(X_train, rowLength, y_train, criteria);
  } 
  auto train_end = std::chrono::high_resolution_clock::now();
  train_duration_count = (train_end - train_start).count();

  std::cout << "Training time: " << train_duration_count << " seconds\n";

  auto eval_start = std::chrono::high_resolution_clock::now();
  score = model.evaluate(X_test, rowLength, y_test);
  auto eval_end = std::chrono::high_resolution_clock::now();
  eval_duration_count = (eval_end - eval_start).count();

  std::cout << "Evaluation time: " << eval_duration_count << " seconds\n";
  std::cout << "Model score with " << loss_func << " : " << score << "\n";
}

inline void
trainAndEvaluateModel(XGBoost &model, const std::vector<double> &X_train,
                      int rowLength, const std::vector<double> &y_train,
                      const std::vector<double> &X_test,
                      const std::vector<double> &y_test, int criteria,
                      double &score, double &train_duration_count,
                      double &eval_duration_count, std::string loss_func) {
  std::cout << "Training process started, please wait...\n";

  auto train_start = std::chrono::high_resolution_clock::now();

  model.train(X_train, rowLength, y_train);

  auto train_end = std::chrono::high_resolution_clock::now();
  train_duration_count = (train_end - train_start).count();

  std::cout << "Training time: " << train_duration_count << " seconds\n";

  auto eval_start = std::chrono::high_resolution_clock::now();
  score = model.evaluate(X_test, rowLength, y_test);
  auto eval_end = std::chrono::high_resolution_clock::now();
  eval_duration_count = (eval_end - eval_start).count();

  std::cout << "Evaluation time: " << eval_duration_count << " seconds\n";
  std::cout << "Model score with " << loss_func << " : " << score << "\n";
}

#endif
