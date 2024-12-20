#include "ensemble_bagging/bagging.h"
#include "ensemble_boosting/boosting.h"
#include "ensemble_boosting/loss_function.h"
#include "ensemble_boosting_XGBoost/boosting_XGBoost.h"
#include "functions_io/functions_io.h"
#include "functions_tree/decision_tree_single.h"
#include "functions_tree/feature_importance.h"
#include "functions_tree/math_functions.h"
#include "functions_tree/tree_visualization.h"
#include "model_comparison/model_comparison.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>

void displayFeatureImportance(
    const std::vector<FeatureImportance::FeatureScore> &scores) {
  std::cout << "\nFeature importance :\n";
  std::cout << std::string(30, '-') << "\n";

  for (const auto &score : scores) {
    std::cout << std::setw(15) << score.feature_name << std::setw(15)
              << std::fixed << std::setprecision(2)
              << score.importance_score * 100.0 << "\n";
  }
  std::cout << std::endl;
}

// input function to set parameters with defaults
template <typename T>
T getInputWithDefault(const std::string &prompt, T defaultValue) {
  std::cout << prompt << " (Default: " << defaultValue << "): ";
  std::string input;
  std::getline(std::cin, input); // Read user input as string

    //If empty return default
  if (input.empty()) {
    return defaultValue;
  }

  
  std::istringstream iss(input);
  T value;
  iss >> value;

  
  if (iss.fail()) {
    std::cerr << "Invalid input. Using default value: " << defaultValue << "\n";
    return defaultValue;
  }
  return value;
}

int main(int argc, char* argv[]) {
  DataIO data_io;
  auto [X, y] = data_io.readCSV("../datasets/cleaned_data.csv");
  if (X.empty() || y.empty()) {
    std::cerr << "Unable to open the data file, please check the path."
              << std::endl;
    return -1;
  }

  // Create folder if non existent
  std::filesystem::path models_dir = "../saved_models";
  if (!std::filesystem::exists(models_dir)) {
      std::filesystem::create_directories(models_dir);
      std::cout << "Directory created: " << models_dir << std::endl;
  }

  // Feature names
  std::vector<std::string> feature_names = {
      "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
      "matrix_size_y"};

  size_t train_size = static_cast<size_t>(X.size() * 0.8);
  std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
  std::vector<double> y_train(y.begin(), y.begin() + train_size);
  std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
  std::vector<double> y_test(y.begin() + train_size, y.end());

  int choice;
  bool use_custom_params = false;
  bool load_request = false; 
  std::string path_model_filename = "";
  std::vector<std::string> params;

  if (argc > 1) {
    choice = std::stoi(argv[1]);
    if (argc > 2 && std::string(argv[2]) == "-p") {
      use_custom_params = true;
      for (int i = 3; i < argc; i++) {
        params.push_back(argv[i]);
      }
    }
    if (argc > 2 && std::string(argv[2]) == "-l") {
      load_request = true;
      path_model_filename = std::string(argv[3]);
    }
  } else {
    //If MainEnsemble does't have any argument
    std::cout << "Choose the method you want to use:\n";
    std::cout << "1: Simple Decision Tree\n";
    std::cout << "2: Bagging\n";
    std::cout << "3: Boosting\n";
    std::cout << "4: Boosting model with XGBoost\n";
    std::cin >> choice;
  }

  if (choice == 1) {
      int maxDepth, minSamplesSplit;
      double minImpurityDecrease;
      int criteria;

      // Create folder if non existent
      std::filesystem::path models_dir = "../saved_models/tree_models";
      if (!std::filesystem::exists(models_dir)) {
          std::filesystem::create_directories(models_dir);
          std::cout << "Directory created: " << models_dir << std::endl;
      }

      if (use_custom_params && params.size() > 3) {
        criteria = std::stoi(params[0]);
        maxDepth = std::stoi(params[1]);
        minSamplesSplit = std::stoi(params[2]);
        minImpurityDecrease = std::stod(params[3]);
      } else if (load_request) {
        DecisionTreeSingle single_tree(0, 0, 0.0, 0); // Temporary
        try {
          single_tree.loadTree(path_model_filename);
          std::cout << "Model loaded successfully from " << path_model_filename << "\n";
        } catch (const std::runtime_error& e) {
          std::cerr << "Error loading the model: " << e.what() << "\n";
          return -1;
        }
        
        // Recover model parameters
        std::map<std::string, std::string> training_params = single_tree.getTrainingParameters();

        // Update parameter variables
        maxDepth = std::stoi(training_params["MaxDepth"]);
        minSamplesSplit = std::stoi(training_params["MinLeafLarge"]);
        minImpurityDecrease = std::stod(training_params["MinError"]);
        criteria = std::stoi(training_params["Criteria"]);

        // Display tree parameters
        std::cout << "Parameters loaded from the model file:\n";
        std::cout << single_tree.getTrainingParametersString() << "\n";
        
        return 0; // Nothing done for the moment but loadable
      } else {
        std::cout << "Generation of default values : " << std::endl
                      << "Default for splitting criteria (MSE)\n"
                      << "Default maximum depth = 60\n"
                      << "Default minimum sample split = 2\n"
                      << "Default minimum impurity decrease = 1e-12\n";
        criteria = 0;
        maxDepth = 60;
        minSamplesSplit = 2;
        minImpurityDecrease = 1e-12;
      }

      std::cout << "Training a single decision tree, please wait...\n";
      DecisionTreeSingle single_tree(maxDepth, minSamplesSplit,
                                    minImpurityDecrease, criteria);

      auto train_start = std::chrono::high_resolution_clock::now();
      single_tree.train(X_train, y_train, criteria);
      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> train_duration = train_end - train_start;
      std::cout << "Training time: " << train_duration.count() << " seconds\n";

      auto eval_start = std::chrono::high_resolution_clock::now();
      // Initialisation pour stocker les résultats de MSE et MAE pour comparer
      double mse_value = 0.0;
      double mae_value = 0.0;
      size_t test_size = X_test.size();
      std::vector<double> y_pred;
      y_pred.reserve(test_size);
      for (const auto &X : X_test) {
        y_pred.push_back(single_tree.predict(X));
      }
      mse_value = Math::computeLossMSE(y_test, y_pred);
      mae_value = Math::computeLossMAE(y_test, y_pred);
      auto eval_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> eval_duration = eval_end - eval_start;

      std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
      std::cout << "Mean Squared Error (MSE): " << mse_value << "\n";
      std::cout << "Mean Absolute Error (MAE): " << mae_value << "\n";

      // computing feature and showing feature importance
      auto feature_importance =
      FeatureImportance::calculateTreeImportance(single_tree, feature_names);
      displayFeatureImportance(feature_importance);

      std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
      int answer = 0;
      std::cin >> answer;
      if (answer == 1) {
        std::cout << "le critère après tout les délires 3: " << criteria << std::endl;
        std::cout << "Please type the name you want to give to the .txt file: \n";
        std::string filename;
        std::cin >> filename;
        std::string path = "../saved_models/tree_models/" + filename;
        std::cout << "Saving tree as: " << filename << "in this path : " << path << std::endl;
        single_tree.saveTree(path);
      }

      // Save results for comparaison
      ModelResults results;
      results.model_name = "Arbre de décision simple";
      results.mse = mse_value;
      results.mae = mse_value;
      results.training_time = train_duration.count();
      results.evaluation_time = eval_duration.count();
      
      // Save parameters
      results.parameters["max_depth"] = maxDepth;
      results.parameters["min_samples_split"] = minSamplesSplit;
      results.parameters["min_impurity_decrease"] = minImpurityDecrease;
      results.parameters["criteria"] = criteria;
      
      // Save characteristic importance
      for (const auto& score : feature_importance) {
          results.feature_importance[score.feature_name] = score.importance_score;
      }
      
      ModelComparison::saveResults(results);

      // Add image for visualization
      std::cout << "Génération de la visualisation de l'arbre..." << std::endl;
      TreeVisualization::generateDotFile(single_tree, "single_tree",
                                        feature_names);
      std::cout << "Visualisation générée dans le dossier 'visualizations'"
                << std::endl;
    
  } else if (choice == 2) {
      int num_trees, max_depth, min_samples_split;
      int criteria;
      int which_loss_func;
      double min_impurity_decrease;

      // Create folder bagging models if non existent
      std::filesystem::path models_dir = "../saved_models/bagging_models";
      if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
        std::cout << "Directory created: " << models_dir << std::endl;
      }

      if (use_custom_params && params.size() > 4) {
        criteria = std::stoi(params[0]);
        which_loss_func = std::stoi(params[1]);
        num_trees = std::stoi(params[2]);
        max_depth = std::stoi(params[3]);
        min_samples_split = std::stoi(params[4]);
        min_impurity_decrease = std::stod(params[5]);
      } else if (load_request) {
        Bagging bagging_model(0, 0, 0, 0.0, nullptr, 0, 0); // Initialisation temporaire

        try {
          bagging_model.load(path_model_filename);
          std::cout << "Model loaded successfully from " << path_model_filename << "\n";
        } catch (const std::runtime_error& e) {
          std::cerr << "Error loading the model: " << e.what() << "\n";
          return -1;
        }

        // Recover model parameters
        std::map<std::string, std::string> training_params = bagging_model.getTrainingParameters();

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
        
        return 0; //Nothing done but model loaded
      } else {
        std::cout << "Generation of default values : " << std::endl
                      << "Default for splitting criteria (MSE)\n"
                      << "Default for comparing trees (MSE)\n"
                      << "Default number of trees to generate : 20\n"
                      << "Default maximum depth = 60\n"
                      << "Default minimum sample split = 2\n"
                      << "Default minimum impurity decrease = 1e-6\n";
        criteria = 0;
        which_loss_func = 0;
        num_trees = 20;
        max_depth = 60;
        min_samples_split = 2;
        min_impurity_decrease = 1e-6;
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
                            min_impurity_decrease, std::move(loss_function), criteria, which_loss_func);

      auto train_start = std::chrono::high_resolution_clock::now();
      bagging_model.train(X_train, y_train, criteria);
      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> train_duration = train_end - train_start;
      std::cout << "Training time (Bagging): " << train_duration.count()
                << " seconds\n";

      auto eval_start = std::chrono::high_resolution_clock::now();
      double mse_or_mae_value = bagging_model.evaluate(X_test, y_test);
      auto eval_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> eval_duration = eval_end - eval_start;
      std::cout << "Evaluation time (Bagging): " << eval_duration.count()
                << " seconds\n";

      std::cout << printMAEorMSE << mse_or_mae_value << "\n";

      // compute and show feature importance
      auto feature_importance = FeatureImportance::calculateBaggingImportance(bagging_model, feature_names);
      displayFeatureImportance(feature_importance);

      // Save model if users wants it
      bool save_model = false;
      std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
      std::cin >> save_model;

      if (save_model) {
          std::string filename;
          std::cout << "Enter the filename to save the model: ";
          std::cin >> filename;
          std::string path = "../saved_models/bagging_models/" + filename;
          bagging_model.save(path);
          std::cout << "Model saved successfully as " << filename << "in this path : " << path << "\n";
      }

      // Save results
      ModelResults results;
      results.model_name = "Bagging";
      results.mse_or_mae = mse_or_mae_value;
      results.training_time = train_duration.count();
      results.evaluation_time = eval_duration.count();
      
      // Save parameters
      results.parameters["n_estimators"] = num_trees;
      results.parameters["max_depth"] = max_depth;
      results.parameters["min_samples_split"] = min_samples_split;
      results.parameters["min_impurity_decrease"] = min_impurity_decrease;
      
      // Save feature importance
      for (const auto& score : feature_importance) {
          results.feature_importance[score.feature_name] = score.importance_score;
      }
      
      ModelComparison::saveResults(results);

      // Add image for visualisation
      std::cout << "Génération des visualisations des arbres..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(bagging_model.getTrees(),
                                                  "bagging", feature_names);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    
  } else if (choice == 3) {
      int n_estimators, max_depth, min_samples_split;
      int criteria;
      int which_loss_func;
      double min_impurity_decrease, learning_rate, initial_prediction;

      // Create boosting folder if new
      std::filesystem::path models_dir = "../saved_models/boosting_models";
      if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
        std::cout << "Directory created: " << models_dir << std::endl;
      }

      if (use_custom_params && params.size() > 5) {
        criteria = std::stoi(params[0]);
        which_loss_func = std::stoi(params[1]);
        n_estimators = std::stoi(params[2]);
        max_depth = std::stoi(params[3]);
        min_samples_split = std::stoi(params[4]);
        min_impurity_decrease = std::stod(params[5]);
        learning_rate = std::stod(params[6]);
      } else if (load_request) {
        Boosting boosting_model(0, 0.0, nullptr, 0, 0, 0.0, 0, 0); // temporary creation

        try {
          boosting_model.load(path_model_filename);
          std::cout << "Model loaded successfully from " << path_model_filename << "\n";
        } catch (const std::runtime_error& e) {
          std::cerr << "Error loading the model: " << e.what() << "\n";
          return -1;
        }

        // Recover model parameters
        std::map<std::string, std::string> training_params = boosting_model.getTrainingParameters();

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
        printMAEorMSE = "Bagging Mean Absolute Error (MAE): ";
      }

      std::cout << "Boosting process started, please wait...\n";
      Boosting boosting_model(n_estimators, learning_rate,
                              std::move(loss_function), max_depth,
                              min_samples_split, min_impurity_decrease, criteria, which_loss_func);

      // model training
      auto train_start = std::chrono::high_resolution_clock::now();
      boosting_model.train(X_train, y_train, criteria);
      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> train_duration = train_end - train_start;
      std::cout << "Training time: " << train_duration.count() << " seconds\n";

      // Model evaluation
      auto eval_start = std::chrono::high_resolution_clock::now();
      double mse_or_mae_value = boosting_model.evaluate(X_test, y_test);
      auto eval_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> eval_duration = eval_end - eval_start;
      std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";

      std::cout << printMAEorMSE << mse_or_mae_value << "\n";

      // Compute and show feature importance
      auto feature_importance = FeatureImportance::calculateBoostingImportance(boosting_model, feature_names);
      displayFeatureImportance(feature_importance);

      // Save model
      bool save_model = false;
      std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
      std::cin >> save_model;
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

      if (save_model) {
          std::string filename;
          std::cout << "Enter the filename to save the model: ";
          std::cin >> filename;
          std::string path = "../saved_models/boosting_models/" + filename;
          boosting_model.save(path);
          std::cout << "Model saved successfully as " << filename << "in this path : " << path << "\n";
      }

      // Save results for comparaison
      ModelResults results;
      results.model_name = "Boosting";
      results.mse_or_mae = mse_or_mae_value;
      results.training_time = train_duration.count();
      results.evaluation_time = eval_duration.count();
      
      // Save features
      results.parameters["n_estimators"] = n_estimators;
      results.parameters["max_depth"] = max_depth;
      results.parameters["min_samples_split"] = min_samples_split;
      results.parameters["min_impurity_decrease"] = min_impurity_decrease;
      results.parameters["learning_rate"] = learning_rate;
      
      for (const auto& score : feature_importance) {
          results.feature_importance[score.feature_name] = score.importance_score;
      }
      
      ModelComparison::saveResults(results);

      // Generate images and save
      std::cout << "Génération des visualisations des arbres..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(boosting_model.getEstimators(), "boosting", feature_names);
      std::cout << "Visualisations générées dans le dossier 'visualizations'" << std::endl;
  } else if (choice == 4) {
        int n_estimators, max_depth;
        int which_loss_func;
        double learning_rate, lambda, alpha, initial_prediction;


        // Create folder if non existent
        std::filesystem::path models_dir = "../saved_models/xgboost_models";
        if (!std::filesystem::exists(models_dir)) {
            std::filesystem::create_directories(models_dir);
            std::cout << "Directory created: " << models_dir << std::endl;
        }

        if (use_custom_params && params.size() >= 5) {
            which_loss_func = std::stoi(params[0]);
            n_estimators = std::stoi(params[1]);
            max_depth = std::stoi(params[2]);
            learning_rate = std::stod(params[3]);
            lambda = std::stod(params[4]);
            alpha = std::stod(params[5]);
        } else if (load_request) {
          XGBoost xgboost_model(0, 0, 0.0, 0.0, 0.0, nullptr, 0); // Initialisation temporaire
          
          try {
            xgboost_model.load(path_model_filename);
            std::cout << "Model loaded successfully from " << path_model_filename << "\n";
          } catch (const std::runtime_error& e) {
            std::cerr << "Error loading the model: " << e.what() << "\n";
            return -1;
          }
          // Recover model parameters
          std::map<std::string, std::string> training_params = xgboost_model.getTrainingParameters();

          // Update parameter variables
          n_estimators = std::stoi(training_params["NumEstimators"]);
          max_depth = std::stoi(training_params["MaxDepth"]);
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
                      << "Default maximum depth = 10\n"
                      << "Default learning rate = 0.1\n"
                      << "Default lambda (L2 regularization) = 1.0\n"
                      << "Default gamma (complexity) = 0.05\n";
            which_loss_func = 0;
            n_estimators = 75;
            max_depth = 10;
            learning_rate = 0.07;
            lambda = 0.3;
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

        std::cout << "Boosting process started, please wait...\n";
        XGBoost xgboost_model(n_estimators, max_depth, learning_rate, lambda, alpha, std::move(loss_function), which_loss_func);

        auto train_start = std::chrono::high_resolution_clock::now();
        xgboost_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training time: " << train_duration.count() << " seconds\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_or_mae_value = xgboost_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;

        std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
        std::cout << printMAEorMSE << mse_or_mae_value << "\n";

        // Compute and show feature importance
        auto feature_importance = xgboost_model.featureImportance(feature_names);
        std::cout << "\nFeature importance:\n";
        std::cout << std::string(30, '-') << "\n";
        for (const auto& [feature, importance] : feature_importance) {
            std::cout << std::setw(15) << feature << std::setw(15)
                     << std::fixed << std::setprecision(2)
                     << importance * 100.0 << "%\n";
        }
        std::cout << std::endl;

        // Save results 
        ModelResults results;
        results.model_name = "XGBoost";
        results.mse_or_mae = mse_or_mae_value;
        results.training_time = train_duration.count();
        results.evaluation_time = eval_duration.count();
        
        // Save parameters
        results.parameters["n_estimators"] = n_estimators;
        results.parameters["max_depth"] = max_depth;
        results.parameters["learning_rate"] = learning_rate;
        results.parameters["lambda"] = lambda;
        results.parameters["gamma"] = alpha;
        
        // Save feature importance
        results.feature_importance = feature_importance;
        
        ModelComparison::saveResults(results);

        std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
        int save_model;
        std::cin >> save_model;

        if (save_model == 1) {
            std::cout << "Enter filename to save the model: ";
            std::string filename;
            std::cin >> filename;
            std::string path = "../saved_models/xgboost_models/" + filename;
            xgboost_model.save(path);
            std::cout << "Model saved successfully as " << filename << "in this path : " << path << "\n";
        }
  } else {
    std::cerr << "Invalid choice! Please choose 1, 2, 3 or 4" << std::endl;
    return -1;
  }
  return 0;
}
