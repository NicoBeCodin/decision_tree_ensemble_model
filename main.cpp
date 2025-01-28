// #include "ensemble_bagging/bagging.h"
// #include "ensemble_boosting/boosting.h"
// #include "ensemble_boosting/loss_function.h"
// #include "ensemble_boosting_XGBoost/boosting_XGBoost.h"
// #include "functions_io/functions_io.h"
// #include "functions_tree/decision_tree_single.h"
// #include "functions_tree/feature_importance.h"
// #include "functions_tree/math_functions.h"
// #include "functions_tree/tree_visualization.h"
// #include "model_comparison/model_comparison.h"
// #include <chrono>
// #include <iomanip>
// #include <iostream>
// #include <string>

// void displayFeatureImportance(
//     const std::vector<FeatureImportance::FeatureScore> &scores) {
//   std::cout << "\nFeature importance :\n";
//   std::cout << std::string(30, '-') << "\n";

//   for (const auto &score : scores) {
//     std::cout << std::setw(15) << score.feature_name << std::setw(15)
//               << std::fixed << std::setprecision(2)
//               << score.importance_score * 100.0 << "\n";
//   }
//   std::cout << std::endl;
// }

// // input function to set parameters with defaults
// template <typename T>
// T getInputWithDefault(const std::string &prompt, T defaultValue) {
//   std::cout << prompt << " (Default: " << defaultValue << "): ";
//   std::string input;
//   std::getline(std::cin, input); // Read user input as string

//     //If empty return default
//   if (input.empty()) {
//     return defaultValue;
//   }

  
//   std::istringstream iss(input);
//   T value;
//   iss >> value;

  
//   if (iss.fail()) {
//     std::cerr << "Invalid input. Using default value: " << defaultValue << "\n";
//     return defaultValue;
//   }
//   return value;
// }

// int main(int argc, char* argv[]) {
//   DataIO data_io;
//   auto [X, y] = data_io.readCSV("../datasets/cleaned_data.csv");
//   if (X.empty() || y.empty()) {
//     std::cerr << "Unable to open the data file, please check the path."
//               << std::endl;
//     return -1;
//   }

//   // Créer le dossier saved_models s'il n'existe pas
//   std::filesystem::path models_dir = "../saved_models";
//   if (!std::filesystem::exists(models_dir)) {
//       std::filesystem::create_directories(models_dir);
//       std::cout << "Directory created: " << models_dir << std::endl;
//   }

//   // Noms des caractéristiques
//   std::vector<std::string> feature_names = {
//       "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
//       "matrix_size_y"};

//   size_t train_size = static_cast<size_t>(X.size() * 0.8);
//   std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
//   std::vector<double> y_train(y.begin(), y.begin() + train_size);
//   std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
//   std::vector<double> y_test(y.begin() + train_size, y.end());

//   int choice;
//   bool use_custom_params = false;
//   std::vector<std::string> params;

//   if (argc > 1) {
//     choice = std::stoi(argv[1]);
//     if (argc > 2 && std::string(argv[2]) == "-p") {
//       use_custom_params = true;
//       for (int i = 3; i < argc; i++) {
//         params.push_back(argv[i]);
//       }
//     }
//   } else {
//     std::cout << "Choose the method you want to use:\n";
//     std::cout << "1: Simple Decision Tree\n";
//     std::cout << "2: Bagging\n";
//     std::cout << "3: Boosting\n";
//     std::cout << "4: Boosting model with XGBoost\n";
//     std::cin >> choice;
//   }

//   if (choice == 1) {
//     DecisionTreeSingle single_tree(0, 0, 0.0); // Initialisation temporaire
//     bool load_existing = false;

//     // Créer le dossier tree_models s'il n'existe pas
//     std::filesystem::path models_dir = "../saved_models/tree_models";
//     if (!std::filesystem::exists(models_dir)) {
//         std::filesystem::create_directories(models_dir);
//         std::cout << "Directory created: " << models_dir << std::endl;
//     }

//     if (!use_custom_params) {
//       std::cout << "Would you like to load an existing tree model? (1 = Yes (for the moment, no use), 0 = No): ";
//       std::cin >> load_existing;
//     }

//     if (load_existing) {
//       std::string model_filename;
//       std::cout << "Enter the filename of the model to load: ";
//       std::cin >> model_filename;

//       std::string path = "../saved_models/tree_models/" + model_filename;

//       try {
//         single_tree.loadTree(path);
//         std::cout << "Model loaded successfully from " << model_filename << "\n";
//       } catch (const std::runtime_error& e) {
//         std::cerr << "Error loading the model: " << e.what() << "\n";
//         return -1;
//       }
//     } else {
//       int maxDepth, minSamplesSplit;
//       double minImpurityDecrease;
//       int criteria = 0;  // Default to MSE

//       if (use_custom_params && params.size() >= 3) {
//         maxDepth = std::stoi(params[0]);
//         minSamplesSplit = std::stoi(params[1]);
//         minImpurityDecrease = std::stod(params[2]);
//       } else {
//         std::cout << "Which method do you want as a splitting criteria: MSE (0) "
//                     "(faster for the moment) or MAE (1) ?"
//                   << std::endl;
//         std::cin >> criteria;

//         std::cout << "You can customize parameters for the decision tree.\n";
//         std::cout << "Press Enter to use the default value or type a new value and "
//                     "press Enter.\n";

//         std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//         maxDepth = getInputWithDefault("Enter maximum depth", 60);
//         minSamplesSplit = getInputWithDefault("Enter minimum number of samples to split", 2);
//         minImpurityDecrease = getInputWithDefault("Enter minimum impurity decrease", 1e-12);
//       }

//       std::cout << "Training a single decision tree, please wait...\n";
//       DecisionTreeSingle single_tree(maxDepth, minSamplesSplit,
//                                     minImpurityDecrease);

//       auto train_start = std::chrono::high_resolution_clock::now();
//       single_tree.train(X_train, y_train, criteria);
//       auto train_end = std::chrono::high_resolution_clock::now();
//       std::chrono::duration<double> train_duration = train_end - train_start;
//       std::cout << "Training time: " << train_duration.count() << " seconds\n";

//       auto eval_start = std::chrono::high_resolution_clock::now();
//       double mse_value = 0.0;
//       size_t test_size = X_test.size();
//       std::vector<double> y_pred;
//       y_pred.reserve(test_size);
//       for (const auto &X : X_test) {
//         y_pred.push_back(single_tree.predict(X));
//       }
//       mse_value = Math::computeLoss(y_test, y_pred);
//       auto eval_end = std::chrono::high_resolution_clock::now();
//       std::chrono::duration<double> eval_duration = eval_end - eval_start;

//       std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
//       std::cout << "Mean Squared Error (MSE): " << mse_value << "\n";

//       // Calcul et affichage de l'importance des caractéristiques
//       auto feature_importance =
//       FeatureImportance::calculateTreeImportance(single_tree, feature_names);
//       displayFeatureImportance(feature_importance);

//       std::cout << "Would you like to save this tree? (0 = no, 1 = yes)\n";
//       int answer = 0;
//       std::cin >> answer;
//       if (answer == 1) {
//         std::cout << "Please type the name you want to give to the .txt file: \n";
//         std::string filename;
//         std::cin >> filename;
//         std::string path = "../saved_models/tree/" + filename;
//         std::cout << "Saving tree as: " << filename << "in this path : " << path << std::endl;
//         single_tree.saveTree(path);
//       }

//       // Sauvegarder les résultats pour la comparaison
//       ModelResults results;
//       results.model_name = "Arbre de décision simple";
//       results.mse = mse_value;
//       results.training_time = train_duration.count();
//       results.evaluation_time = eval_duration.count();
      
//       // Sauvegarder les paramètres
//       results.parameters["max_depth"] = maxDepth;
//       results.parameters["min_samples_split"] = minSamplesSplit;
//       results.parameters["min_impurity_decrease"] = minImpurityDecrease;
      
//       // Sauvegarder l'importance des caractéristiques
//       for (const auto& score : feature_importance) {
//           results.feature_importance[score.feature_name] = score.importance_score;
//       }
      
//       ModelComparison::saveResults(results);

//       // Ajout de la visualisation
//       std::cout << "Génération de la visualisation de l'arbre..." << std::endl;
//       TreeVisualization::generateDotFile(single_tree, "single_tree",
//                                         feature_names);
//       std::cout << "Visualisation générée dans le dossier 'visualizations'"
//                 << std::endl;
//     }
//   } else if (choice == 2) {
//     Bagging bagging_model(0, 0, 0, 0.0);
//     bool load_existing = false;

//     // Créer le dossier bagging_models s'il n'existe pas
//     std::filesystem::path models_dir = "../saved_models/bagging_models";
//     if (!std::filesystem::exists(models_dir)) {
//         std::filesystem::create_directories(models_dir);
//         std::cout << "Directory created: " << models_dir << std::endl;
//     }

//     if (!use_custom_params) {
//       std::cout << "Would you like to load an existing model? (1 = Yes (for the moment, no use), 0 = No): ";
//       std::cin >> load_existing;
//     }

//     if (load_existing) {
//       std::string model_filename;
//       std::cout << "Enter the filename of the model to load: ";
//       std::cin >> model_filename;

//       std::string path = "../saved_models/bagging_models/" + model_filename;

//       try {
//         bagging_model.load(model_filename);
//         std::cout << "Model loaded successfully from " << model_filename << "\n";
//       } catch (const std::runtime_error& e) {
//         std::cerr << "Error loading the model: " << e.what() << "\n";
//         return -1;
//       }
//     } else {
//       int num_trees, max_depth, min_samples_split;
//       double min_impurity_decrease;

//       if (use_custom_params && params.size() >= 4) {
//         num_trees = std::stoi(params[0]);
//         max_depth = std::stoi(params[1]);
//         min_samples_split = std::stoi(params[2]);
//         min_impurity_decrease = std::stod(params[3]);
//       } else {
//         std::cout << "You can customize parameters for the bagging process and it's trees\n";
//         std::cout << "Press Enter to use the default value or type a new value and "
//                     "press Enter.\n";

//         num_trees = getInputWithDefault("Enter number of trees to generate", 20);
//         max_depth = getInputWithDefault("Enter max depth", 60);
//         min_samples_split = getInputWithDefault("Enter minimum samples to split", 2);
//         min_impurity_decrease = getInputWithDefault("Enter minimum impurity decrease", 1e-6);
//       }

//       std::cout << "Bagging process started, please wait...\n";
//       Bagging bagging_model(num_trees, max_depth, min_samples_split,
//                             min_impurity_decrease);

//       auto train_start = std::chrono::high_resolution_clock::now();
//       bagging_model.train(X_train, y_train);
//       auto train_end = std::chrono::high_resolution_clock::now();
//       std::chrono::duration<double> train_duration = train_end - train_start;
//       std::cout << "Training time (Bagging): " << train_duration.count()
//                 << " seconds\n";

//       auto eval_start = std::chrono::high_resolution_clock::now();
//       double mse_value = bagging_model.evaluate(X_test, y_test);
//       auto eval_end = std::chrono::high_resolution_clock::now();
//       std::chrono::duration<double> eval_duration = eval_end - eval_start;
//       std::cout << "Evaluation time (Bagging): " << eval_duration.count()
//                 << " seconds\n";

//       std::cout << "Bagging Mean Squared Error (MSE): " << mse_value << "\n";

//       // Calcul et affichage de l'importance des caractéristiques pour le bagging
//       auto feature_importance = FeatureImportance::calculateBaggingImportance(bagging_model, feature_names);
//       displayFeatureImportance(feature_importance);

//       // Sauvegarde du modèle si l'utilisateur le souhaite
//       bool save_model = false;
//       std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
//       std::cin >> save_model;

//       if (save_model) {
//           std::string filename;
//           std::cout << "Enter the filename to save the model: ";
//           std::cin >> filename;
//           std::string path = "../saved_models/bagging_models/" + filename;
//           bagging_model.save(path);
//           std::cout << "Model saved successfully as " << filename << "in this path : " << path << "\n";
//       }

//       // Sauvegarder les résultats pour la comparaison
//       ModelResults results;
//       results.model_name = "Bagging";
//       results.mse = mse_value;
//       results.training_time = train_duration.count();
//       results.evaluation_time = eval_duration.count();
      
//       // Sauvegarder les paramètres
//       results.parameters["n_estimators"] = num_trees;
//       results.parameters["max_depth"] = max_depth;
//       results.parameters["min_samples_split"] = min_samples_split;
//       results.parameters["min_impurity_decrease"] = min_impurity_decrease;
      
//       // Sauvegarder l'importance des caractéristiques
//       for (const auto& score : feature_importance) {
//           results.feature_importance[score.feature_name] = score.importance_score;
//       }
      
//       ModelComparison::saveResults(results);

//       // Ajout de la visualisation
//       std::cout << "Génération des visualisations des arbres..." << std::endl;
//       TreeVisualization::generateEnsembleDotFiles(bagging_model.getTrees(),
//                                                   "bagging", feature_names);
//       std::cout << "Visualisations générées dans le dossier 'visualizations'"
//                 << std::endl;
//     }
//   } else if (choice == 3) {
//     Boosting boosting_model(0, 0.0, nullptr, 0, 0, 0.0);
//     bool load_existing = false;

//     // Créer le dossier boosting_models s'il n'existe pas
//     std::filesystem::path models_dir = "../saved_models/boosting_models";
//     if (!std::filesystem::exists(models_dir)) {
//         std::filesystem::create_directories(models_dir);
//         std::cout << "Directory created: " << models_dir << std::endl;
//     }

//     if (!use_custom_params) {
//       std::cout << "Would you like to load an existing model? (1 = Yes, 0 = No): ";
//       std::cin >> load_existing;
//     }

//     if (load_existing) {
//       std::string model_filename;
//       std::cout << "Enter the filename of the model to load: ";
//       std::cin >> model_filename;

//       std::string path = "../saved_models/boosting_models/" + model_filename;

//       try {
//         boosting_model.load(path);
//         std::cout << "Model loaded successfully from " << model_filename << "\n";
//       } catch (const std::runtime_error& e) {
//         std::cerr << "Error loading the model: " << e.what() << "\n";
//         return -1;
//       }
//     } else {
//       int n_estimators, max_depth, min_samples_split;
//       double min_impurity_decrease, learning_rate;

//       if (use_custom_params && params.size() >= 5) {
//         n_estimators = std::stoi(params[0]);
//         max_depth = std::stoi(params[1]);
//         min_samples_split = std::stoi(params[2]);
//         min_impurity_decrease = std::stod(params[3]);
//         learning_rate = std::stod(params[4]);
//       } else {
//         std::cout << "You can customize parameters for the boosting process and it's trees\n";
//         std::cout << "Press Enter to use the default value or type a new value and "
//                     "press Enter.\n";

//         n_estimators = getInputWithDefault("Enter number of estimators", 75);
//         max_depth = getInputWithDefault("Enter max depth", 15);
//         min_samples_split = getInputWithDefault("Enter minimum sample split", 3);
//         min_impurity_decrease = getInputWithDefault("Enter minimum impurity decrease", 1e-5);
//         learning_rate = getInputWithDefault("Enter learning rate", 0.07);
//       }

//       std::cout << "Boosting process started, please wait...\n";
//       auto loss_function = std::make_unique<LeastSquaresLoss>();
//       Boosting boosting_model(n_estimators, learning_rate,
//                               std::move(loss_function), max_depth,
//                               min_samples_split, min_impurity_decrease);

//       // Entraînement du modèle
//       auto train_start = std::chrono::high_resolution_clock::now();
//       boosting_model.train(X_train, y_train);
//       auto train_end = std::chrono::high_resolution_clock::now();
//       std::chrono::duration<double> train_duration = train_end - train_start;
//       std::cout << "Training time: " << train_duration.count() << " seconds\n";

//       // Évaluation du modèle
//       auto eval_start = std::chrono::high_resolution_clock::now();
//       double mse_value = boosting_model.evaluate(X_test, y_test);
//       auto eval_end = std::chrono::high_resolution_clock::now();
//       std::chrono::duration<double> eval_duration = eval_end - eval_start;
//       std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";

//       std::cout << "Boosting Mean Squared Error (MSE): " << mse_value << "\n";

//       // Calcul et affichage de l'importance des caractéristiques pour le boosting
//       auto feature_importance = FeatureImportance::calculateBoostingImportance(boosting_model, feature_names);
//       displayFeatureImportance(feature_importance);

//       // Sauvegarde du modèle si l'utilisateur le souhaite
//       bool save_model = false;
//       std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
//       std::cin >> save_model;

//       if (save_model) {
//           std::string filename;
//           std::cout << "Enter the filename to save the model: ";
//           std::cin >> filename;
//           std::string path = "../saved_models/boosting_models/" + filename;
//           boosting_model.save(path);
//           std::cout << "Model saved successfully as " << filename << "in this path : " << path << "\n";
//       }

//       // Sauvegarder les résultats pour la comparaison
//       ModelResults results;
//       results.model_name = "Boosting";
//       results.mse = mse_value;
//       results.training_time = train_duration.count();
//       results.evaluation_time = eval_duration.count();
      
//       // Sauvegarder les paramètres
//       results.parameters["n_estimators"] = n_estimators;
//       results.parameters["max_depth"] = max_depth;
//       results.parameters["min_samples_split"] = min_samples_split;
//       results.parameters["min_impurity_decrease"] = min_impurity_decrease;
//       results.parameters["learning_rate"] = learning_rate;
      
//       // Sauvegarder l'importance des caractéristiques
//       for (const auto& score : feature_importance) {
//           results.feature_importance[score.feature_name] = score.importance_score;
//       }
      
//       ModelComparison::saveResults(results);

//       // Ajout de la visualisation
//       std::cout << "Génération des visualisations des arbres..." << std::endl;
//       TreeVisualization::generateEnsembleDotFiles(boosting_model.getEstimators(), "boosting", feature_names);
//       std::cout << "Visualisations générées dans le dossier 'visualizations'" << std::endl;
//     }
//   } else if (choice == 4) {
//     XGBoost* xgboost_model = nullptr;
//     bool load_existing = false;

//     // Créer le dossier xgboost_models s'il n'existe pas
//     std::filesystem::path models_dir = "../saved_models/xgboost_models";
//     if (!std::filesystem::exists(models_dir)) {
//         std::filesystem::create_directories(models_dir);
//         std::cout << "Directory created: " << models_dir << std::endl;
//     }

//     if (!use_custom_params) {
//         std::cout << "Would you like to load an existing model? (1 = Yes, 0 = No): ";
//         std::cin >> load_existing;
//     }

//     if (load_existing) {
//         std::string model_filename;
//         std::cout << "Enter the filename of the model to load: ";
//         std::cin >> model_filename;

//         std::string path = "../saved_models/xgboost_models/" + model_filename;

//         try {
//             // Load model logic here
//             std::cout << "Model loaded successfully from " << model_filename << "\n";
//         } catch (const std::runtime_error& e) {
//             std::cerr << "Error loading the model: " << e.what() << "\n";
//             return -1;
//         }
//     } else {
//         int n_estimators, max_depth;
//         double learning_rate, lambda, gamma;

//         if (use_custom_params && params.size() >= 5) {
//             n_estimators = std::stoi(params[0]);
//             max_depth = std::stoi(params[1]);
//             learning_rate = std::stod(params[2]);
//             lambda = std::stod(params[3]);
//             gamma = std::stod(params[4]);
//         } else {
//             std::cout << "You can customize parameters for XGBoost\n";
//             std::cout << "Press Enter to use the default value or type a new value and press Enter.\n";

//             std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//             n_estimators = getInputWithDefault("Enter number of estimators", 10);
//             max_depth = getInputWithDefault("Enter max depth", 5);
//             learning_rate = getInputWithDefault("Enter learning rate", 0.1);
//             lambda = getInputWithDefault("Enter lambda (L2 regularization)", 1.0);
//             gamma = getInputWithDefault("Enter gamma (complexity)", 0.0);
//         }

//         std::cout << "Boosting process started, please wait...\n";
//         auto loss_function = std::make_unique<LeastSquaresLoss>();
//         xgboost_model = new XGBoost(n_estimators, max_depth, learning_rate, lambda, gamma, std::move(loss_function));

//         auto train_start = std::chrono::high_resolution_clock::now();
//         xgboost_model->train(X_train, y_train);
//         auto train_end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> train_duration = train_end - train_start;
//         std::cout << "Training time: " << train_duration.count() << " seconds\n";

//         auto eval_start = std::chrono::high_resolution_clock::now();
//         double mse_value = xgboost_model->evaluate(X_test, y_test);
//         auto eval_end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> eval_duration = eval_end - eval_start;

//         std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
//         std::cout << "Boosting Mean Squared Error (MSE): " << mse_value << "\n";

//         // Calcul et affichage de l'importance des caractéristiques
//         auto feature_importance = xgboost_model->featureImportance(feature_names);
//         std::cout << "\nFeature importance:\n";
//         std::cout << std::string(30, '-') << "\n";
//         for (const auto& [feature, importance] : feature_importance) {
//             std::cout << std::setw(15) << feature << std::setw(15)
//                      << std::fixed << std::setprecision(2)
//                      << importance * 100.0 << "%\n";
//         }
//         std::cout << std::endl;

//         // Sauvegarder les résultats pour la comparaison
//         ModelResults results;
//         results.model_name = "XGBoost";
//         results.mse = mse_value;
//         results.training_time = train_duration.count();
//         results.evaluation_time = eval_duration.count();
        
//         // Sauvegarder les paramètres
//         results.parameters["n_estimators"] = n_estimators;
//         results.parameters["max_depth"] = max_depth;
//         results.parameters["learning_rate"] = learning_rate;
//         results.parameters["lambda"] = lambda;
//         results.parameters["gamma"] = gamma;
        
//         // Sauvegarder l'importance des caractéristiques
//         results.feature_importance = feature_importance;
        
//         ModelComparison::saveResults(results);

//         std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
//         int save_model;
//         std::cin >> save_model;

//         if (save_model == 1) {
//             std::cout << "Enter filename to save the model: ";
//             std::string filename;
//             std::cin >> filename;
//             std::string path = "../saved_models/xgboost_models/" + filename;
//             xgboost_model->save(path);
//             std::cout << "Model saved as: " << filename << std::endl;
//         }
//     }

//     if (xgboost_model != nullptr) {
//         delete xgboost_model;
//     }
//   } else {
//     std::cerr << "Invalid choice! Please choose 1, 2, 3 or 4" << std::endl;
//     return -1;
//   }

//   return 0;
// }



//Linear version


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
#include <memory>
#include <string>

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
  int rowLength = 11;
  auto [X, y] = data_io.readCSV("../datasets/sample_400_rows.csv", rowLength);
  if (X.empty() || y.empty()) {
    std::cerr << "Unable to open the data file, please check the path."
              << std::endl;
    return -1;
  }

  std::cout<<"X size : "<<X.size()<<std::endl;
  std::cout<<"y size : "<<y.size()<<std::endl;
  

  // Créer le dossier saved_models s'il n'existe pas
  std::filesystem::path models_dir = "../saved_models";
  if (!std::filesystem::exists(models_dir)) {
      std::filesystem::create_directories(models_dir);
      std::cout << "Directory created: " << models_dir << std::endl;
  }

  // Noms des caractéristiques
  std::vector<std::string> feature_names = {
      "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
      "matrix_size_y"};

  //We resize rowLength because that it the size of a data row without label
  rowLength = rowLength-1;
  size_t train_size = static_cast<size_t>(y.size() * 0.8) * rowLength;
  
  std::cout<<"Train size : "<<train_size<<std::endl;

  std::vector<double> X_train(X.begin(), X.begin() + train_size);
  std::vector<double> y_train(y.begin(), y.begin() + train_size/10);
  std::vector<double> X_test(X.begin() + train_size, X.end());
  std::vector<double> y_test(y.begin() + train_size /10, y.end());

  std::cout<<"X_train size : "<<X_train.size()<<std::endl;
  std::cout<<"y_train size : "<<y_train.size()<<std::endl;
  std::cout<<"X_test size : "<<X_test.size()<<std::endl;
  std::cout<<"y_test size : "<<y_test.size()<< "\n"<<std::endl;

  
  int choice;
  bool use_custom_params = false;
  std::vector<std::string> params;

  if (argc > 1) {
    choice = std::stoi(argv[1]);
    if (argc > 2 && std::string(argv[2]) == "-p") {
      use_custom_params = true;
      for (int i = 3; i < argc; i++) {
        params.push_back(argv[i]);
      }
    }
  } else {
    std::cout << "Choose the method you want to use:\n";
    std::cout << "1: Simple Decision Tree\n";
    std::cout << "2: Bagging\n";
    std::cout << "3: Boosting\n";
    std::cout << "4: Boosting model with XGBoost\n";
    std::cin >> choice;
  }

  if (choice == 1) {
    DecisionTreeSingle single_tree(0, 0, 0.0); // Initialisation temporaire
    bool load_existing = false;

    // Créer le dossier tree_models s'il n'existe pas
    std::filesystem::path models_dir = "../saved_models/tree_models";
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
        std::cout << "Directory created: " << models_dir << std::endl;
    }

    if (!use_custom_params) {
      std::cout << "Would you like to load an existing tree model? (1 = Yes (for the moment, no use), 0 = No): ";
      std::cin >> load_existing;
    }

    if (load_existing) {
      std::string model_filename;
      std::cout << "Enter the filename of the model to load: ";
      std::cin >> model_filename;

      std::string path = "../saved_models/tree_models/" + model_filename;

      try {
        single_tree.loadTree(path);
        std::cout << "Model loaded successfully from " << model_filename << "\n";
      } catch (const std::runtime_error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }
    } else {
      int maxDepth, minSamplesSplit;
      double minImpurityDecrease;
      int criteria = 0;  // Default to MSE

      if (use_custom_params && params.size() >= 3) {
        maxDepth = std::stoi(params[0]);
        minSamplesSplit = std::stoi(params[1]);
        minImpurityDecrease = std::stod(params[2]);
      } else {
        std::cout << "Which method do you want as a splitting criteria: MSE (0) "
                    "(faster for the moment) or MAE (1) ?"
                  << std::endl;
        std::cin >> criteria;

        std::cout << "You can customize parameters for the decision tree.\n";
        std::cout << "Press Enter to use the default value or type a new value and "
                    "press Enter.\n";

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        maxDepth = getInputWithDefault("Enter maximum depth", 60);
        minSamplesSplit = getInputWithDefault("Enter minimum number of samples to split", 2);
        minImpurityDecrease = getInputWithDefault("Enter minimum impurity decrease", 1e-12);
      }

      std::cout << "Training a single decision tree, please wait...\n";
      DecisionTreeSingle single_tree(maxDepth, minSamplesSplit,
                                    minImpurityDecrease);

      auto train_start = std::chrono::high_resolution_clock::now();
      single_tree.train(X_train,rowLength, y_train, criteria);
      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> train_duration = train_end - train_start;
      std::cout << "Training time: " << train_duration.count() << " seconds\n";

      auto eval_start = std::chrono::high_resolution_clock::now();
      double mse_value = 0.0;
      size_t test_size = X_test.size();
      std::vector<double> y_pred;
      y_pred.reserve(test_size);
      for (size_t i = 0; i<y_test.size(); ++i) {
        std::vector<double> sample(X_test.begin()+ i*rowLength, X_test.begin() + (i+1)*rowLength);
        y_pred.push_back(single_tree.predict(sample));
      }
      mse_value = Math::computeLossMSE(y_test, y_pred);
      auto eval_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> eval_duration = eval_end - eval_start;

      std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
      std::cout << "Mean Squared Error (MSE): " << mse_value << "\n";

      // Calcul et affichage de l'importance des caractéristiques
      auto feature_importance =
      FeatureImportance::calculateTreeImportance(single_tree, feature_names);
      displayFeatureImportance(feature_importance);

      std::cout << "Would you like to save this tree? (0 = no, 1 = yes)\n";
      int answer = 0;
      std::cin >> answer;
      if (answer == 1) {
        std::cout << "Please type the name you want to give to the .txt file: \n";
        std::string filename;
        std::cin >> filename;
        std::string path = "../saved_models/tree/" + filename;
        std::cout << "Saving tree as: " << filename << "in this path : " << path << std::endl;
        single_tree.saveTree(path);
      }

      // Sauvegarder les résultats pour la comparaison
      ModelResults results;
      results.model_name = "Arbre de décision simple";
      results.mse = mse_value;
      results.training_time = train_duration.count();
      results.evaluation_time = eval_duration.count();
      
      // Sauvegarder les paramètres
      results.parameters["max_depth"] = maxDepth;
      results.parameters["min_samples_split"] = minSamplesSplit;
      results.parameters["min_impurity_decrease"] = minImpurityDecrease;
      
      // Sauvegarder l'importance des caractéristiques
      for (const auto& score : feature_importance) {
          results.feature_importance[score.feature_name] = score.importance_score;
      }
      
      ModelComparison::saveResults(results);

      // Ajout de la visualisation
      std::cout << "Génération de la visualisation de l'arbre..." << std::endl;
      TreeVisualization::generateDotFile(single_tree, "single_tree",
                                        feature_names);
      std::cout << "Visualisation générée dans le dossier 'visualizations'"
                << std::endl;
    }
  } else if (choice == 2) {
    Bagging bagging_model(0, 0, 0, 0.0);
    bool load_existing = false;

    // Créer le dossier bagging_models s'il n'existe pas
    std::filesystem::path models_dir = "../saved_models/bagging_models";
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
        std::cout << "Directory created: " << models_dir << std::endl;
    }

    if (!use_custom_params) {
      std::cout << "Would you like to load an existing model? (1 = Yes (for the moment, no use), 0 = No): ";
      std::cin >> load_existing;
    }

    if (load_existing) {
      std::string model_filename;
      std::cout << "Enter the filename of the model to load: ";
      std::cin >> model_filename;

      std::string path = "../saved_models/bagging_models/" + model_filename;

      try {
        bagging_model.load(model_filename);
        std::cout << "Model loaded successfully from " << model_filename << "\n";
      } catch (const std::runtime_error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }
    } else {
      int num_trees, max_depth, min_samples_split;
      double min_impurity_decrease;

      if (use_custom_params && params.size() >= 4) {
        num_trees = std::stoi(params[0]);
        max_depth = std::stoi(params[1]);
        min_samples_split = std::stoi(params[2]);
        min_impurity_decrease = std::stod(params[3]);
      } else {
        std::cout << "You can customize parameters for the bagging process and it's trees\n";
        std::cout << "Press Enter to use the default value or type a new value and "
                    "press Enter.\n";

        num_trees = getInputWithDefault("Enter number of trees to generate", 20);
        max_depth = getInputWithDefault("Enter max depth", 60);
        min_samples_split = getInputWithDefault("Enter minimum samples to split", 2);
        min_impurity_decrease = getInputWithDefault("Enter minimum impurity decrease", 1e-6);
      }

      std::cout << "Bagging process started, please wait...\n";
      auto loss_function = std::make_unique<LeastSquaresLoss>();
      Bagging bagging_model(num_trees, max_depth, min_samples_split,
                            min_impurity_decrease, std::move(loss_function));

      std::cout<<"Finished initializing the bagging model, starting training..."<<std::endl;

      auto train_start = std::chrono::high_resolution_clock::now();
      bagging_model.train(X_train, rowLength, y_train);
      std::cout<<"Training finished"<<std::endl;

      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> train_duration = train_end - train_start;
      std::cout << "Training time (Bagging): " << train_duration.count()
                << " seconds\n";

      auto eval_start = std::chrono::high_resolution_clock::now();
      std::cout<<"Starting evaluation..."<<std::endl;
      double mse_value = bagging_model.evaluate(X_test, rowLength, y_test);
      auto eval_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> eval_duration = eval_end - eval_start;
      std::cout << "Evaluation time (Bagging): " << eval_duration.count()
                << " seconds\n";

      std::cout << "Bagging Mean Squared Error (MSE): " << mse_value << "\n";

      // Calcul et affichage de l'importance des caractéristiques pour le bagging
      auto feature_importance = FeatureImportance::calculateBaggingImportance(bagging_model, feature_names);
      displayFeatureImportance(feature_importance);

      // Sauvegarde du modèle si l'utilisateur le souhaite
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

      // Sauvegarder les résultats pour la comparaison
      ModelResults results;
      results.model_name = "Bagging";
      results.mse = mse_value;
      results.training_time = train_duration.count();
      results.evaluation_time = eval_duration.count();
      
      // Sauvegarder les paramètres
      results.parameters["n_estimators"] = num_trees;
      results.parameters["max_depth"] = max_depth;
      results.parameters["min_samples_split"] = min_samples_split;
      results.parameters["min_impurity_decrease"] = min_impurity_decrease;
      
      // Sauvegarder l'importance des caractéristiques
      for (const auto& score : feature_importance) {
          results.feature_importance[score.feature_name] = score.importance_score;
      }
      
      ModelComparison::saveResults(results);

      // Ajout de la visualisation
      std::cout << "Génération des visualisations des arbres..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(bagging_model.getTrees(),
                                                  "bagging", feature_names);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    }
  } else if (choice == 3) {
    Boosting boosting_model(0, 0.0, nullptr, 0, 0, 0.0);
    bool load_existing = false;

    // Créer le dossier boosting_models s'il n'existe pas
    std::filesystem::path models_dir = "../saved_models/boosting_models";
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
        std::cout << "Directory created: " << models_dir << std::endl;
    }

    if (!use_custom_params) {
      std::cout << "Would you like to load an existing model? (1 = Yes, 0 = No): ";
      std::cin >> load_existing;
    }

    if (load_existing) {
      std::string model_filename;
      std::cout << "Enter the filename of the model to load: ";
      std::cin >> model_filename;

      std::string path = "../saved_models/boosting_models/" + model_filename;

      try {
        boosting_model.load(path);
        std::cout << "Model loaded successfully from " << model_filename << "\n";
      } catch (const std::runtime_error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
      }
    } else {
      int n_estimators, max_depth, min_samples_split;
      double min_impurity_decrease, learning_rate;

      if (use_custom_params && params.size() >= 5) {
        n_estimators = std::stoi(params[0]);
        max_depth = std::stoi(params[1]);
        min_samples_split = std::stoi(params[2]);
        min_impurity_decrease = std::stod(params[3]);
        learning_rate = std::stod(params[4]);
      } else {
        std::cout << "You can customize parameters for the boosting process and it's trees\n";
        std::cout << "Press Enter to use the default value or type a new value and "
                    "press Enter.\n";

        n_estimators = getInputWithDefault("Enter number of estimators", 75);
        max_depth = getInputWithDefault("Enter max depth", 15);
        min_samples_split = getInputWithDefault("Enter minimum sample split", 3);
        min_impurity_decrease = getInputWithDefault("Enter minimum impurity decrease", 1e-5);
        learning_rate = getInputWithDefault("Enter learning rate", 0.07);
      }

      std::cout << "Boosting process started, please wait...\n";
      auto loss_function = std::make_unique<LeastSquaresLoss>();
      Boosting boosting_model(n_estimators, learning_rate,
                              std::move(loss_function), max_depth,
                              min_samples_split, min_impurity_decrease);

      // Entraînement du modèle
      auto train_start = std::chrono::high_resolution_clock::now();
      boosting_model.train(X_train, rowLength, y_train, 0);
      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> train_duration = train_end - train_start;
      std::cout << "Training time: " << train_duration.count() << " seconds\n";

      // Évaluation du modèle
      auto eval_start = std::chrono::high_resolution_clock::now();
      double mse_value = boosting_model.evaluate(X_test, rowLength, y_test);
      auto eval_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> eval_duration = eval_end - eval_start;
      std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";

      std::cout << "Boosting Mean Squared Error (MSE): " << mse_value << "\n";

      // Calcul et affichage de l'importance des caractéristiques pour le boosting
      auto feature_importance = FeatureImportance::calculateBoostingImportance(boosting_model, feature_names);
      displayFeatureImportance(feature_importance);

      // Sauvegarde du modèle si l'utilisateur le souhaite
      bool save_model = false;
      std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
      std::cin >> save_model;

      if (save_model) {
          std::string filename;
          std::cout << "Enter the filename to save the model: ";
          std::cin >> filename;
          std::string path = "../saved_models/boosting_models/" + filename;
          boosting_model.save(path);
          std::cout << "Model saved successfully as " << filename << "in this path : " << path << "\n";
      }

      // Sauvegarder les résultats pour la comparaison
      ModelResults results;
      results.model_name = "Boosting";
      results.mse = mse_value;
      results.training_time = train_duration.count();
      results.evaluation_time = eval_duration.count();
      
      // Sauvegarder les paramètres
      results.parameters["n_estimators"] = n_estimators;
      results.parameters["max_depth"] = max_depth;
      results.parameters["min_samples_split"] = min_samples_split;
      results.parameters["min_impurity_decrease"] = min_impurity_decrease;
      results.parameters["learning_rate"] = learning_rate;
      
      // Sauvegarder l'importance des caractéristiques
      for (const auto& score : feature_importance) {
          results.feature_importance[score.feature_name] = score.importance_score;
      }
      
      ModelComparison::saveResults(results);

      // Ajout de la visualisation
      std::cout << "Génération des visualisations des arbres..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(boosting_model.getEstimators(), "boosting", feature_names);
      std::cout << "Visualisations générées dans le dossier 'visualizations'" << std::endl;
    }
  } else if (choice == 4) {
    XGBoost* xgboost_model = nullptr;
    bool load_existing = false;

    // Créer le dossier xgboost_models s'il n'existe pas
    std::filesystem::path models_dir = "../saved_models/xgboost_models";
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
        std::cout << "Directory created: " << models_dir << std::endl;
    }

    if (!use_custom_params) {
        std::cout << "Would you like to load an existing model? (1 = Yes, 0 = No): ";
        std::cin >> load_existing;
    }

    if (load_existing) {
        std::string model_filename;
        std::cout << "Enter the filename of the model to load: ";
        std::cin >> model_filename;

        std::string path = "../saved_models/xgboost_models/" + model_filename;

        try {
            // Load model logic here
            std::cout << "Model loaded successfully from " << model_filename << "\n";
        } catch (const std::runtime_error& e) {
            std::cerr << "Error loading the model: " << e.what() << "\n";
            return -1;
        }
    } else {
        int n_estimators, max_depth;
        double learning_rate, lambda, gamma;

        if (use_custom_params && params.size() >= 5) {
            n_estimators = std::stoi(params[0]);
            max_depth = std::stoi(params[1]);
            learning_rate = std::stod(params[2]);
            lambda = std::stod(params[3]);
            gamma = std::stod(params[4]);
        } else {
            std::cout << "You can customize parameters for XGBoost\n";
            std::cout << "Press Enter to use the default value or type a new value and press Enter.\n";

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            n_estimators = getInputWithDefault("Enter number of estimators", 10);
            max_depth = getInputWithDefault("Enter max depth", 5);
            learning_rate = getInputWithDefault("Enter learning rate", 0.1);
            lambda = getInputWithDefault("Enter lambda (L2 regularization)", 1.0);
            gamma = getInputWithDefault("Enter gamma (complexity)", 0.0);
        }

        std::cout << "Boosting process started, please wait...\n";
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        xgboost_model = new XGBoost(n_estimators, max_depth, learning_rate, lambda, gamma, std::move(loss_function));

        auto train_start = std::chrono::high_resolution_clock::now();
        xgboost_model->train(X_train, rowLength, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training time: " << train_duration.count() << " seconds\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = xgboost_model->evaluate(X_test, rowLength, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;

        std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
        std::cout << "Boosting Mean Squared Error (MSE): " << mse_value << "\n";

        // Calcul et affichage de l'importance des caractéristiques
        auto feature_importance = xgboost_model->featureImportance(feature_names);
        std::cout << "\nFeature importance:\n";
        std::cout << std::string(30, '-') << "\n";
        for (const auto& [feature, importance] : feature_importance) {
            std::cout << std::setw(15) << feature << std::setw(15)
                     << std::fixed << std::setprecision(2)
                     << importance * 100.0 << "%\n";
        }
        std::cout << std::endl;

        // Sauvegarder les résultats pour la comparaison
        ModelResults results;
        results.model_name = "XGBoost";
        results.mse = mse_value;
        results.training_time = train_duration.count();
        results.evaluation_time = eval_duration.count();
        
        // Sauvegarder les paramètres
        results.parameters["n_estimators"] = n_estimators;
        results.parameters["max_depth"] = max_depth;
        results.parameters["learning_rate"] = learning_rate;
        results.parameters["lambda"] = lambda;
        results.parameters["gamma"] = gamma;
        
        // Sauvegarder l'importance des caractéristiques
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
            xgboost_model->save(path);
            std::cout << "Model saved as: " << filename << std::endl;
        }
    }

    if (xgboost_model != nullptr) {
        delete xgboost_model;
    }
  } else {
    std::cerr << "Invalid choice! Please choose 1, 2, 3 or 4" << std::endl;
    return -1;
  }

  return 0;
}
