#include "run_models.h"

void runSingleDecisionTreeModel(DecisionTreeParams params, DataParams data_params) {
    std::cout << "Training a single decision tree, please wait...\n";
  
    // Feature names
    std::vector<std::string> feature_names = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x", "matrix_size_y"};
    
    DecisionTreeSingle single_tree(params.maxDepth, params.minSamplesSplit,
                                   params.minImpurityDecrease, params.criteria, params.numThreads, params.useOmp);
  
    auto train_start = std::chrono::high_resolution_clock::now();
    single_tree.train(data_params.X_train, data_params.rowLength, data_params.y_train, params.criteria);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_duration = std::chrono::duration_cast<std::chrono::duration<double>>(train_end - train_start).count();
    std::cout << "Training time: " << train_duration << " seconds\n";
  
    auto eval_start = std::chrono::high_resolution_clock::now();
    // Initialisation pour stocker les résultats de MSE et MAE pour comparer
    double mse_value = 0.0;
    double mae_value = 0.0;
    single_tree.evaluate(data_params.X_test, data_params.rowLength, data_params.y_test, mse_value, mae_value);
  
    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_duration = std::chrono::duration_cast<std::chrono::duration<double>>(eval_end - eval_start).count();
  
    std::cout << "Evaluation time: " << eval_duration << " seconds\n";
    std::cout << "Mean Squared Error (MSE): " << mse_value << "\n";
    std::cout << "Mean Absolute Error (MAE): " << mae_value << "\n";
  
    // computing feature and showing feature importance
    auto feature_importance = FeatureImportance::calculateTreeImportance(single_tree, feature_names);
    displayFeatureImportance(feature_importance);
  
    // Save model or not
    saveModel(single_tree);
  
    // Save results for comparaison
    ModelResults results;
    results.model_name = "Arbre de décision simple";
    results.mse = mse_value;
    results.mae = mae_value;
    results.training_time = train_duration;
    results.evaluation_time = eval_duration;
  
    // Save parameters
    results.parameters["max_depth"] = params.maxDepth;
    results.parameters["min_samples_split"] = params.minSamplesSplit;
    results.parameters["min_impurity_decrease"] = params.minImpurityDecrease;
    results.parameters["criteria"] = params.criteria;
  
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
                << (params.criteria == 0 ? "MSE" : "MAE") << "..." << std::endl;
      TreeVisualization::generateDotFile(single_tree, "single_tree",
                                         feature_names, params.criteria);
      std::cout << "Visualisation générée dans le dossier 'visualizations'"
                << std::endl;
    }
}

void runBaggingModel(BaggingParams params, DataParams data_params) {
    std::cout << "Training a Bagging model, please wait...\n";
  
    // Feature names
    std::vector<std::string> feature_names = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x", "matrix_size_y"};
  
    std::unique_ptr<LossFunction> loss_function;
    std::string printMAEorMSE;
  
    if (params.whichLossFunction == 0) {
      loss_function = std::make_unique<LeastSquaresLoss>();
      printMAEorMSE = "Bagging Mean Squared Error (MSE): ";
    } else {
      loss_function = std::make_unique<MeanAbsoluteLoss>();
      printMAEorMSE = "Bagging Mean Absolute Error (MAE): ";
    }
  
    std::cout << "Bagging process started, please wait...\n";
  
    Bagging bagging_model(params.numTrees, params.maxDepth, params.minSamplesSplit,
                          params.minImpurityDecrease, std::move(loss_function),
                          params.criteria, params.whichLossFunction, params.numThreads);
  
    double score = 0.0;
    double train_duration_count = 0.0;
    double evaluation_duration_count = 0.0;
  
    trainAndEvaluateModel(bagging_model, data_params.X_train, data_params.rowLength, data_params.y_train, data_params.X_test,
                          data_params.y_test, params.criteria, score, train_duration_count,
                          evaluation_duration_count, printMAEorMSE);
  
    // compute and show feature importance
    auto feature_importance = FeatureImportance::calculateBaggingImportance(bagging_model, feature_names);
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
    results.parameters["n_estimators"] = params.numTrees;
    results.parameters["max_depth"] = params.maxDepth;
    results.parameters["min_samples_split"] = params.minSamplesSplit;
    results.parameters["min_impurity_decrease"] = params.minImpurityDecrease;
  
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
                << (params.criteria == 0 ? "MSE" : "MAE") << "..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(
          bagging_model.getTrees(), "bagging", feature_names, params.criteria);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    }
}

void runBoostingModel(BoostingParams params, DataParams data_params) {
    std::cout << "Training a Boosting model, please wait...\n";
  
    // Feature names
    std::vector<std::string> feature_names = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x", "matrix_size_y"};
  
    std::unique_ptr<LossFunction> loss_function;
    std::string printMAEorMSE;
  
    if (params.whichLossFunction == 0) {
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
  
    Boosting boosting_model(params.nEstimators, params.learningRate, std::move(loss_function), 
                            params.maxDepth, params.minSamplesSplit, params.minImpurityDecrease, 
                            params.criteria, params.whichLossFunction);
  
    trainAndEvaluateModel(boosting_model, data_params.X_train, data_params.rowLength, data_params.y_train, 
                          data_params.X_test, data_params.y_test, params.criteria, score, train_duration_count,
                          eval_duration_count, printMAEorMSE);
  
    // Compute and show feature importance
    auto feature_importance = FeatureImportance::calculateBoostingImportance(boosting_model, feature_names);
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
    results.parameters["n_estimators"] = params.nEstimators;
    results.parameters["max_depth"] = params.maxDepth;
    results.parameters["min_samples_split"] = params.minSamplesSplit;
    results.parameters["min_impurity_decrease"] = params.minImpurityDecrease;
    results.parameters["learning_rate"] = params.learningRate;
  
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
                << (params.criteria == 0 ? "MSE" : "MAE") << "..." << std::endl;
      TreeVisualization::generateEnsembleDotFiles(
          boosting_model.getEstimators(), "boosting", feature_names, params.criteria);
      std::cout << "Visualisations générées dans le dossier 'visualizations'"
                << std::endl;
    }
}