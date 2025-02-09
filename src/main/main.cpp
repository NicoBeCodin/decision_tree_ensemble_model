#include "../ensemble/bagging/bagging.h"
#include "../ensemble/boosting/boosting.h"
#include "../ensemble/boosting_XGBoost/boosting_XGBoost.h"
#include "../functions/feature/feature_importance.h"
#include "../functions/io/functions_io.h"
#include "../functions/math/math_functions.h"
#include "../functions/tree/decision_tree_single.h"
#include "../functions/tree/vizualization/tree_visualization.h"
#include "../model_comparison/model_comparison.h"
#include "utility.h"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {

  ProgramOptions programOptions = parseCommandLineArguments(argc, argv);

  DataIO data_io;
  int rowLength = 11;
  auto [X, y] =
      data_io.readCSV("../datasets/processed/cleaned_data.csv", rowLength);
  if (X.empty() || y.empty()) {
    std::cerr << "Unable to open the data file, please check the path."
              << std::endl;
    return -1;
  }

  std::cout << "X size : " << X.size() << std::endl;
  std::cout << "y size : " << y.size() << std::endl;

  // Creates saved models folder if non existant
  createDirectory("../saved_models");

  // Feature names
  std::vector<std::string> feature_names = {
      "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
      "matrix_size_y"};

  // We resize rowLength because that it the size of a data row without label
  rowLength = rowLength - 1;
  size_t train_size = static_cast<size_t>(y.size() * 0.8) * rowLength;

  std::cout << "Train size : " << train_size << std::endl;

  std::vector<double> X_train(X.begin(), X.begin() + train_size);
  std::vector<double> y_train(y.begin(), y.begin() + train_size / 10);
  std::vector<double> X_test(X.begin() + train_size, X.end());
  std::vector<double> y_test(y.begin() + train_size / 10, y.end());

  std::cout << "X_train size : " << X_train.size() << std::endl;
  std::cout << "y_train size : " << y_train.size() << std::endl;
  std::cout << "X_test size : " << X_test.size() << std::endl;
  std::cout << "y_test size : " << y_test.size() << "\n" << std::endl;


  if (programOptions.choice == 1) {
    int maxDepth, minSamplesSplit;
    double minImpurityDecrease;
    int criteria;
    int numThreads;

    // Create folder if non existent
    createDirectory("../saved_models/tree_models");

    if (programOptions.use_custom_params && programOptions.params.size() > 3) {
      criteria = std::stoi(programOptions.params[0]);
      maxDepth = std::stoi(programOptions.params[1]);
      minSamplesSplit = std::stoi(programOptions.params[2]);
      minImpurityDecrease = std::stod(programOptions.params[3]);
      numThreads = std::stoi(programOptions.params[4]);
      // This is to make sure it's a power of two
      numThreads = adjustNumThreads(numThreads);

    } else if (programOptions.load_request) {
      DecisionTreeSingle single_tree(0, 0, 0.0, 0); // Temporary
      try {
        single_tree.loadTree(programOptions.path_model_filename);
        std::cout << "Model loaded successfully from "
                  << programOptions.path_model_filename << "\n";
      } catch (const std::runtime_error &e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }

      // Recover model parameters
      std::map<std::string, std::string> training_params =
          single_tree.getTrainingParameters();

      // Update parameter variables
      maxDepth = std::stoi(training_params["MaxDepth"]);
      minSamplesSplit = std::stoi(training_params["MinLeafLarge"]);
      minImpurityDecrease = std::stod(training_params["MinError"]);
      criteria = std::stoi(training_params["Criteria"]);

      // Retrieve numThreads safely with a default value of 1, for
      // Retrocompatibility
      numThreads = (training_params.find("NumThreads") != training_params.end())
                       ? std::stoi(training_params["NumThreads"])
                       : 1;

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << single_tree.getTrainingParametersString() << "\n";

      return 0; // Nothing done for the moment but loadable
    } else {
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)\n"
                << "Default maximum depth = 60\n"
                << "Default minimum sample split = 2\n"
                << "Default minimum impurity decrease = 1e-12\n"
                << "Default number of threads is 1";
      criteria = 0;
      maxDepth = 60;
      minSamplesSplit = 2;
      minImpurityDecrease = 1e-12;
      numThreads = 1;
    }

    std::cout << "Training a single decision tree, please wait...\n";
    DecisionTreeSingle single_tree(maxDepth, minSamplesSplit,
                                   minImpurityDecrease, criteria, numThreads);

    auto train_start = std::chrono::high_resolution_clock::now();
    single_tree.train(X_train, rowLength, y_train, criteria);
    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end - train_start;
    std::cout << "Training time: " << train_duration.count() << " seconds\n";

    auto eval_start = std::chrono::high_resolution_clock::now();
    // Initialisation pour stocker les résultats de MSE et MAE pour comparer
    double mse_value = 0.0;
    double mae_value = 0.0;
    single_tree.evaluate(X_test, rowLength, y_test, mse_value, mae_value);

    auto eval_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eval_duration = eval_end - eval_start;

    std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
    std::cout << "Mean Squared Error (MSE): " << mse_value << "\n";
    std::cout << "Mean Absolute Error (MAE): " << mae_value << "\n";

    // computing feature and showing feature importance
    auto feature_importance =
        FeatureImportance::calculateTreeImportance(single_tree, feature_names);
    displayFeatureImportance(feature_importance);

    // Save model or not
    saveModel(single_tree);

    // Save results for comparaison
    ModelResults results;
    results.model_name = "Arbre de décision simple";
    results.mse = mse_value;
    results.mae = mae_value;
    results.training_time = train_duration.count();
    results.evaluation_time = eval_duration.count();

    // Save parameters
    results.parameters["max_depth"] = maxDepth;
    results.parameters["min_samples_split"] = minSamplesSplit;
    results.parameters["min_impurity_decrease"] = minImpurityDecrease;
    results.parameters["criteria"] = criteria;

    // Save characteristic importance
    for (const auto &score : feature_importance) {
      results.feature_importance[score.feature_name] = score.importance_score;
    }

    ModelComparison::saveResults(results);

    // Generate visualisation if users wants it
    bool visualisation_ask = false;
    std::cout << "Would you like to generate a visualisation of this model? (1 "
                 "= Yes, 0 = No): ";
    std::cin >> visualisation_ask;

    if (visualisation_ask) {
      // Add image for visualization
      std::cout << "Génération de la visualisation de l'arbre avec critère: "
                << (criteria == 0 ? "MSE" : "MAE") << "..." << std::endl;
      TreeVisualization::generateDotFile(single_tree, "single_tree",
                                         feature_names, criteria);
      std::cout << "Visualisation générée dans le dossier 'visualizations'"
                << std::endl;
    }
  } else if (programOptions.choice == 2) {
    int num_trees, max_depth, min_samples_split;
    int criteria;
    int which_loss_func;
    double min_impurity_decrease;
    int numThreads;

    createDirectory("../saved_models/bagging_models");

    if (programOptions.use_custom_params && programOptions.params.size() > 4) {
      criteria = std::stoi(programOptions.params[0]);
      which_loss_func = std::stoi(programOptions.params[1]);
      num_trees = std::stoi(programOptions.params[2]);
      max_depth = std::stoi(programOptions.params[3]);
      min_samples_split = std::stoi(programOptions.params[4]);
      min_impurity_decrease = std::stod(programOptions.params[5]);
      numThreads = std::stoi(programOptions.params[6]);
    } else if (programOptions.load_request) {
      Bagging bagging_model(0, 0, 0, 0.0, nullptr, 0, 0, 1); // Temp init

      try {
        bagging_model.load(programOptions.path_model_filename);
        std::cout << "Model loaded successfully from "
                  << programOptions.path_model_filename << "\n";
      } catch (const std::runtime_error &e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }

      // Recover model parameters
      std::map<std::string, std::string> training_params =
          bagging_model.getTrainingParameters();

      // Update parameter variables
      num_trees = std::stoi(training_params["NumTrees"]);
      max_depth = std::stoi(training_params["MaxDepth"]);
      min_samples_split = std::stoi(training_params["MinSamplesSplit"]);
      min_impurity_decrease = std::stod(training_params["MinImpurityDecrease"]);
      criteria = std::stoi(training_params["Criteria"]);
      which_loss_func = std::stoi(training_params["WhichLossFunction"]);

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << bagging_model.getTrainingParametersString() << "\n";

      return 0; // Nothing done but model loaded
    } else {
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)\n"
                << "Default for comparing trees (MSE)\n"
                << "Default number of trees to generate : 20\n"
                << "Default maximum depth = 60\n"
                << "Default minimum sample split = 2\n"
                << "Default minimum impurity decrease = 1e-6\n"
                << "Default amount of threads used is 1\n";
      criteria = 0;
      which_loss_func = 0;
      num_trees = 20;
      max_depth = 60;
      min_samples_split = 2;
      min_impurity_decrease = 1e-6;
      numThreads = 1;
    }

    std::unique_ptr<LossFunction> loss_function;
    std::string printMAEorMSE;

    if (which_loss_func == 0) {
      loss_function = std::make_unique<LeastSquaresLoss>();
      printMAEorMSE = "Bagging Mean Squared Error (MSE): ";
    } else {
      loss_function = std::make_unique<MeanAbsoluteLoss>();
      printMAEorMSE = "Bagging Mean Absolute Error (MAE): ";
    }

    std::cout << "Bagging process started, please wait...\n";

    Bagging bagging_model(num_trees, max_depth, min_samples_split,
                          min_impurity_decrease, std::move(loss_function),
                          criteria, which_loss_func, numThreads);

    double score = 0.0;
    double train_duration_count = 0.0;
    double evaluation_duration_count = 0.0;

    trainAndEvaluateModel(bagging_model, X_train, rowLength, y_train, X_test,
                          y_test, criteria, score, train_duration_count,
                          evaluation_duration_count, printMAEorMSE);

    // compute and show feature importance
    auto feature_importance = FeatureImportance::calculateBaggingImportance(
        bagging_model, feature_names);
    displayFeatureImportance(feature_importance);

    // Save model if users wants it
    saveModel(bagging_model);

    // Save results
    ModelResults results;
    results.model_name = "Bagging";
    results.mse_or_mae = score;
    results.training_time = train_duration_count;
    results.evaluation_time = evaluation_duration_count;

    // Save parameters
    results.parameters["n_estimators"] = num_trees;
    results.parameters["max_depth"] = max_depth;
    results.parameters["min_samples_split"] = min_samples_split;
    results.parameters["min_impurity_decrease"] = min_impurity_decrease;

    // Save feature importance
    for (const auto &score : feature_importance) {
      results.feature_importance[score.feature_name] = score.importance_score;
    }

    ModelComparison::saveResults(results);

    // Generate visualisation if users wants it
    bool visualisation_ask = false;
    std::cout << "Would you like to genarate a visualisation of this model? (1 "
                 "= Yes, 0 = No): ";
    std::cin >> visualisation_ask;

    if (visualisation_ask) {
      // Add image for visualisation
      std::cout << "Génération de la visualisation des arbres avec critère: "
                << (criteria == 0 ? "MSE" : "MAE") << "..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(
          bagging_model.getTrees(), "bagging", feature_names, criteria);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    }
  } else if (programOptions.choice == 3) {
    int n_estimators, max_depth, min_samples_split;
    int criteria;
    int which_loss_func;
    double min_impurity_decrease, learning_rate, initial_prediction;

    // Create boosting folder if new
    createDirectory("../saved_models/boosting_models");

    if (programOptions.use_custom_params && programOptions.params.size() > 5) {
      criteria = std::stoi(programOptions.params[0]);
      which_loss_func = std::stoi(programOptions.params[1]);
      n_estimators = std::stoi(programOptions.params[2]);
      max_depth = std::stoi(programOptions.params[3]);
      min_samples_split = std::stoi(programOptions.params[4]);
      min_impurity_decrease = std::stod(programOptions.params[5]);
      learning_rate = std::stod(programOptions.params[6]);
    } else if (programOptions.load_request) {
      Boosting boosting_model(0, 0.0, nullptr, 0, 0, 0.0, 0,
                              0); // temporary creation

      try {
        boosting_model.load(programOptions.path_model_filename);
        std::cout << "Model loaded successfully from "
                  << programOptions.path_model_filename << "\n";
      } catch (const std::runtime_error &e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }

      // Recover model parameters
      std::map<std::string, std::string> training_params =
          boosting_model.getTrainingParameters();

      // Update parameter variables
      n_estimators = std::stoi(training_params["NumEstimators"]);
      learning_rate = std::stod(training_params["LearningRate"]);
      max_depth = std::stoi(training_params["MaxDepth"]);
      min_samples_split = std::stoi(training_params["MinSamplesSplit"]);
      min_impurity_decrease = std::stod(training_params["MinImpurityDecrease"]);
      initial_prediction = std::stod(training_params["InitialPrediction"]);
      criteria = std::stoi(training_params["Criteria"]);
      which_loss_func = std::stoi(training_params["WhichLossFunction"]);

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << boosting_model.getTrainingParametersString() << "\n";

      return 0; // Nothing done but loadable
    } else {
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)\n"
                << "Default for comparing trees (MSE)\n"
                << "Default number of estimators : 75\n"
                << "Default maximum depth = 15\n"
                << "Default minimum sample split = 3\n"
                << "Default minimum impurity decrease = 1e-5\n"
                << "Default learning rate = 0.07\n";
      criteria = 0;
      which_loss_func = 0;
      n_estimators = 75;
      max_depth = 15;
      min_samples_split = 3;
      min_impurity_decrease = 1e-5;
      learning_rate = 0.07;
    }

    std::unique_ptr<LossFunction> loss_function;
    std::string printMAEorMSE;

    if (which_loss_func == 0) {
      loss_function = std::make_unique<LeastSquaresLoss>();
      printMAEorMSE = "Boosting Mean Square Error (MSE): ";
    } else {
      loss_function = std::make_unique<MeanAbsoluteLoss>();
      printMAEorMSE = "Boosting Mean Absolute Error (MAE): ";
    }

    double score = 0.0;
    double train_duration_count = 0.0;
    double eval_duration_count = 0.0;

    std::cout << "Boosting process started, please wait...\n";

    Boosting boosting_model(
        n_estimators, learning_rate, std::move(loss_function), max_depth,
        min_samples_split, min_impurity_decrease, criteria, which_loss_func);

    trainAndEvaluateModel(
        boosting_model, X_train, rowLength,
        y_train, X_test,
        y_test, criteria, score,
        train_duration_count,  eval_duration_count, printMAEorMSE);

    // Compute and show feature importance
    auto feature_importance = FeatureImportance::calculateBoostingImportance(
        boosting_model, feature_names);
    displayFeatureImportance(feature_importance);

    // Save model
    saveModel(boosting_model);

    // Save results for comparaison
    ModelResults results;
    results.model_name = "Boosting";
    results.mse_or_mae = score;
    results.training_time = train_duration_count;
    results.evaluation_time = eval_duration_count;

    // Save features
    results.parameters["n_estimators"] = n_estimators;
    results.parameters["max_depth"] = max_depth;
    results.parameters["min_samples_split"] = min_samples_split;
    results.parameters["min_impurity_decrease"] = min_impurity_decrease;
    results.parameters["learning_rate"] = learning_rate;

    for (const auto &score : feature_importance) {
      results.feature_importance[score.feature_name] = score.importance_score;
    }

    ModelComparison::saveResults(results);

    // Generate visualisation if users wants it
    bool visualisation_ask = false;
    std::cout << "Would you like to genarate a visualisation of this model? (1 "
                 "= Yes, 0 = No): ";
    std::cin >> visualisation_ask;

    if (visualisation_ask) {
      // Generate images and save
      std::cout << "Génération de la visualisation des arbres avec critère: "
                << (criteria == 0 ? "MSE" : "MAE") << "..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(
          boosting_model.getEstimators(), "boosting", feature_names, criteria);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    }
  } else if (programOptions.choice == 4) {
    int n_estimators, max_depth, min_samples_split;
    int which_loss_func;
    double learning_rate, lambda, alpha, initial_prediction;

    // Create folder if non existent
    createDirectory("../saved_models/xgboost_models");

    if (programOptions.use_custom_params && programOptions.params.size() > 5) {
      which_loss_func = std::stoi(programOptions.params[0]);
      n_estimators = std::stoi(programOptions.params[1]);
      max_depth = std::stoi(programOptions.params[2]);
      min_samples_split = std::stoi(programOptions.params[3]);
      learning_rate = std::stod(programOptions.params[4]);
      lambda = std::stod(programOptions.params[5]);
      alpha = std::stod(programOptions.params[6]);
    } else if (programOptions.load_request) {
      XGBoost xgboost_model(0, 0, 0, 0.0, 0.0, 0.0, nullptr,
                            0); // Initialisation temporaire

      try {
        xgboost_model.load(programOptions.path_model_filename);
        std::cout << "Model loaded successfully from "
                  << programOptions.path_model_filename << "\n";
      } catch (const std::runtime_error &e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }
      // Recover model parameters
      std::map<std::string, std::string> training_params =
          xgboost_model.getTrainingParameters();

      // Update parameter variables
      n_estimators = std::stoi(training_params["NumEstimators"]);
      max_depth = std::stoi(training_params["MaxDepth"]);
      min_samples_split = std::stoi(training_params["MinSamplesSplit"]);
      learning_rate = std::stod(training_params["LearningRate"]);
      lambda = std::stod(training_params["Lambda"]);
      alpha = std::stod(training_params["Alpha"]);
      initial_prediction = std::stod(training_params["InitialPrediction"]);
      which_loss_func = std::stoi(training_params["WhichLossFunction"]);

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << xgboost_model.getTrainingParametersString() << "\n";

      return 0; // Nothing done for the moment but loadable
    } else {
      std::cout << "Generation of default values : " << std::endl
                << "Default for comparing trees (MSE)\n"
                << "Default number of estimators : 75\n"
                << "Default maximum depth = 15\n"
                << "Default minimum sample split = 3\n"
                << "Default learning rate = 0.1\n"
                << "Default lambda (L2 regularization) = 1.0\n"
                << "Default gamma (complexity) = 0.05\n";
      which_loss_func = 0;
      n_estimators = 75;
      max_depth = 15;
      min_samples_split = 3;
      learning_rate = 0.07;
      lambda = 1.0;
      alpha = 0.05;
    }

    std::unique_ptr<LossFunction> loss_function;
    std::string printMAEorMSE;

    if (which_loss_func == 0) {
      loss_function = std::make_unique<LeastSquaresLoss>();
      printMAEorMSE = "Boosting (XGBoost) Mean Square Error (MSE): ";
    } else if (which_loss_func == 1) {
      loss_function = std::make_unique<MeanAbsoluteLoss>();
      printMAEorMSE = "Boosting (XGBoost) Mean Absolute Error (MAE): ";
    }

    double score = 0.0;
    double train_duration_count= 0.0;
    double eval_duration_count = 0.0;

    std::cout << "Boosting process started, please wait...\n";
    XGBoost xgboost_model(n_estimators, max_depth, min_samples_split,
                          learning_rate, lambda, alpha,
                          std::move(loss_function), which_loss_func);


    trainAndEvaluateModel(xgboost_model, X_train, rowLength, y_train, X_test, y_test, -1, score, train_duration_count, eval_duration_count, printMAEorMSE);

    // Compute and show feature importance
    auto feature_importance = xgboost_model.featureImportance(feature_names);
    std::cout << "\nFeature importance:\n";
    std::cout << std::string(30, '-') << "\n";
    for (const auto &[feature, importance] : feature_importance) {
      std::cout << std::setw(15) << feature << std::setw(15) << std::fixed
                << std::setprecision(2) << importance * 100.0 << "%\n";
    }
    std::cout << std::endl;

    // Save results
    ModelResults results;
    results.model_name = "XGBoost";
    results.mse_or_mae = score;
    results.training_time = train_duration_count;
    results.evaluation_time = eval_duration_count;

    // Save parameters
    results.parameters["n_estimators"] = n_estimators;
    results.parameters["max_depth"] = max_depth;
    results.parameters["learning_rate"] = learning_rate;
    results.parameters["lambda"] = lambda;
    results.parameters["gamma"] = alpha;

    // Save feature importance
    results.feature_importance = feature_importance;

    ModelComparison::saveResults(results);

    saveModel(xgboost_model);
    // Generate visualisation if users wants it
    bool visualisation_ask = false;
    std::cout << "Would you like to genarate a visualisation of this model? (1 "
                 "= Yes, 0 = No): ";
    std::cin >> visualisation_ask;

    if (visualisation_ask) {
      // Generate images and save
      std::cout << "Génération de la visualisation des arbres..." << std::endl;
      TreeVisualization::generateEnsembleDotFilesXGBoost(
          xgboost_model.getEstimators(), "xgboost", feature_names);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    }
  } else {
    std::cerr << "Invalid choice! Please choose 1, 2, 3 or 4" << std::endl;
    return -1;
  }
  return 0;
}
