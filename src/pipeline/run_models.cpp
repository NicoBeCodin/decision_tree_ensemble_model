#include "run_models.h"
static void to_matrix(const std::vector<double>& flat,
  int n_cols,
  std::vector<std::vector<double>>& out)
{
int n_rows = flat.size() / n_cols;
out.assign(n_rows, std::vector<double>(n_cols));
for (int i = 0; i < n_rows; ++i)
for (int j = 0; j < n_cols; ++j)
out[i][j] = flat[i * n_cols + j];
}

void runSingleDecisionTreeModel(DecisionTreeParams params,
                                DataParams data_params) {
  std::cout << "Training a single decision tree, please wait...\n";

  // Feature names
  std::vector<std::string> feature_names = {
      "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
      "matrix_size_y"};

  DecisionTreeSingle single_tree(params.maxDepth, params.minSamplesSplit,
                                 params.minImpurityDecrease, params.criteria,
                                 params.useOMP, params.numThreads);

  auto train_start = std::chrono::high_resolution_clock::now();
  single_tree.train(data_params.X_train, data_params.rowLength,
                    data_params.y_train, params.criteria);
  auto train_end = std::chrono::high_resolution_clock::now();
  double train_duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(train_end -
                                                                train_start)
          .count();
  std::cout << "Training time: " << train_duration << " seconds\n";

  auto eval_start = std::chrono::high_resolution_clock::now();
  // Initialisation pour stocker les résultats de MSE et MAE pour comparer
  double mse_value = 0.0;
  double mae_value = 0.0;
  single_tree.evaluate(data_params.X_test, data_params.rowLength,
                       data_params.y_test, mse_value, mae_value);

  auto eval_end = std::chrono::high_resolution_clock::now();
  double eval_duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(eval_end -
                                                                eval_start)
          .count();

  std::cout << "Evaluation time: " << eval_duration << " seconds\n";
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
  results.training_time = train_duration;
  results.evaluation_time = eval_duration;

  // Save parameters
  results.parameters["max_depth"] = params.maxDepth;
  results.parameters["min_samples_split"] = params.minSamplesSplit;
  results.parameters["min_impurity_decrease"] = params.minImpurityDecrease;
  results.parameters["criteria"] = params.criteria;
  results.parameters["use_omp"] = params.useOMP ? 1.0 : 0.0;
  results.parameters["num_threads"] = params.numThreads;

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
  int mpiFinaliseLater = 0; // ← declare outside #ifdef

  int mpiRank = 0, mpiSize = 1;
#ifdef USE_MPI

  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#endif
  // Feature names
  std::vector<std::string> feature_names = {
      "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
      "matrix_size_y"};

  std::unique_ptr<LossFunction> loss_function;
  std::string printMAEorMSE;

  if (mpiRank == 0) {

    std::cout << "Training a Bagging model, please wait...\n";

    if (params.whichLossFunction == 0) {
      loss_function = std::make_unique<LeastSquaresLoss>();
      printMAEorMSE = "Bagging Mean Squared Error (MSE): ";
    } else {
      loss_function = std::make_unique<MeanAbsoluteLoss>();
      printMAEorMSE = "Bagging Mean Absolute Error (MAE): ";
    }
    if (mpiRank == 0) {
      std::cout << "Bagging process started, please wait...\n";
    }
  }

  Bagging bagging_model(params.numTrees, params.maxDepth,
                        params.minSamplesSplit, params.minImpurityDecrease,
                        std::move(loss_function), params.criteria,
                        params.whichLossFunction, params.useOMP, params.numThreads);

  double score = 0.0;
  double train_duration_count = 0.0;
  double evaluation_duration_count = 0.0;

  trainAndEvaluateModel(bagging_model, data_params.X_train,
                        data_params.rowLength, data_params.y_train,
                        data_params.X_test, data_params.y_test, params.criteria,
                        score, train_duration_count, evaluation_duration_count,
                        printMAEorMSE, mpiRank);

#ifdef USE_MPI
  /* make sure everyone finishes before we touch the model */
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (mpiRank == 0) {
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
    results.parameters["n_estimators"] = params.numTrees;
    results.parameters["max_depth"] = params.maxDepth;
    results.parameters["min_samples_split"] = params.minSamplesSplit;
    results.parameters["min_impurity_decrease"] = params.minImpurityDecrease;
    results.parameters["criteria"] = params.criteria;
    results.parameters["which_loss_function"] = params.whichLossFunction;
    results.parameters["use_omp"] = params.useOMP ? 1.0 : 0.0;
    results.parameters["num_threads"] = params.numThreads;

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
#ifdef USE_MPI
  if (mpiFinaliseLater)
    MPI_Finalize();
#endif
}

void runBoostingModel(BoostingParams params, DataParams data_params) {
  std::cout << "Training a Boosting model, please wait...\n";

  // Feature names
  std::vector<std::string> feature_names = {
      "p1",           "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x",
      "matrix_size_y"};

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

  Boosting boosting_model(
      params.nEstimators, params.learningRate, std::move(loss_function),
      params.maxDepth, params.minSamplesSplit, params.minImpurityDecrease,
      params.criteria, params.whichLossFunction, params.useOMP, params.numThreads);

  trainAndEvaluateModel(boosting_model, data_params.X_train,
                        data_params.rowLength, data_params.y_train,
                        data_params.X_test, data_params.y_test, params.criteria,
                        score, train_duration_count, eval_duration_count,
                        printMAEorMSE, 0);

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
  results.parameters["n_estimators"] = params.nEstimators;
  results.parameters["max_depth"] = params.maxDepth;
  results.parameters["min_samples_split"] = params.minSamplesSplit;
  results.parameters["min_impurity_decrease"] = params.minImpurityDecrease;
  results.parameters["learning_rate"] = params.learningRate;
  results.parameters["initial_prediction"] = boosting_model.getInitialPrediction();
  results.parameters["criteria"] = params.criteria;
  results.parameters["which_loss_function"] = params.whichLossFunction;
  results.parameters["use_omp"] = params.useOMP ? 1.0 : 0.0;
  results.parameters["num_threads"] = params.numThreads;

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
    TreeVisualization::generateEnsembleDotFiles(boosting_model.getEstimators(),
                                                "boosting", feature_names,
                                                params.criteria);
    std::cout << "Visualisations générées dans le dossier 'visualizations'"
              << std::endl;
  }
}

void runLightGBMModel(const LightGBMParams &params,
                      const DataParams &data_params) {
  std::cout << "============= LightGBM Model Training ============="
            << std::endl;

  int n_samples_train = static_cast<int>(data_params.y_train.size());
  int n_features = data_params.rowLength;
  std::vector<std::vector<float>> X_train(n_samples_train,
                                          std::vector<float>(n_features));
  for (int i = 0; i < n_samples_train; ++i)
    for (int j = 0; j < n_features; ++j)
      X_train[i][j] =
          static_cast<float>(data_params.X_train[i * n_features + j]);

  std::vector<float> y_train(n_samples_train);
  for (int i = 0; i < n_samples_train; ++i)
    y_train[i] = static_cast<float>(data_params.y_train[i]);

  std::ostringstream oss;
  oss << "objective=regression" << " num_leaves=" << params.numLeaves
      << " max_depth=" << params.maxDepth
      << " learning_rate=" << params.learningRate
      << " bagging_fraction=" << params.subsample
      << " feature_fraction=" << params.colsampleBytree;
  std::string lgbParamsStr = oss.str();

  MyLightGBM model;
  auto t0 = std::chrono::high_resolution_clock::now();
  model.train(X_train, y_train, lgbParamsStr, params.nEstimators);
  auto t1 = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[LightGBM] Training time: " << train_time << " s" << std::endl;

  int n_samples_test = static_cast<int>(data_params.y_test.size());

  std::vector<float> X_test_flat(data_params.X_test.begin(),
                                 data_params.X_test.end());
  auto t2 = std::chrono::high_resolution_clock::now();
  std::vector<double> preds =
      model.predict(X_test_flat, n_samples_test, n_features);
  auto t3 = std::chrono::high_resolution_clock::now();
  double predict_time = std::chrono::duration<double>(t3 - t2).count();
  std::cout << "[LightGBM] Prediction time: " << predict_time << " s"
            << std::endl;

  double mse = 0.0, mae = 0.0;
  for (int i = 0; i < n_samples_test; ++i) {
    double diff = data_params.y_test[i] - preds[i];
    mse += diff * diff;
    mae += std::abs(diff);
  }
  mse /= n_samples_test;
  mae /= n_samples_test;
  std::cout << "[LightGBM] MSE = " << mse << ", MAE = " << mae << std::endl;

  auto importances = model.featureImportance();
  std::cout << "[LightGBM] Feature importance:" << std::endl;
  for (int i = 0; i < static_cast<int>(importances.size()); ++i) {
    std::cout << "  F" << i << ": " << importances[i] << std::endl;
  }

  std::string model_dir = "../saved_models/lightgbm_models";
  createDirectory(model_dir);
  std::string model_file = model_dir + "/lightgbm.model";
  model.saveModel(model_file);
  std::cout << "Model saved to: " << model_file << std::endl;

  std::string imp_file = model_dir + "/feature_importance.txt";
  std::ofstream ofs(imp_file);
  for (double v : importances)
    ofs << v << "\n";
  ofs.close();
  std::cout << "Feature importance saved to: " << imp_file << std::endl;

  ModelResults results;
  results.model_name = "LightGBM";
  results.mse_or_mae = mse;
  results.training_time = train_time;
  results.evaluation_time = predict_time;
  results.parameters["n_estimators"] = params.nEstimators;
  results.parameters["learning_rate"] = params.learningRate;
  results.parameters["max_depth"] = params.maxDepth;
  results.parameters["num_leaves"] = params.numLeaves;
  results.parameters["subsample"] = params.subsample;
  results.parameters["colsample_bytree"] = params.colsampleBytree;
  for (int i = 0; i < static_cast<int>(importances.size()); ++i)
      results.feature_importance["F" + std::to_string(i)] = importances[i];
  ModelComparison::saveResults(results);

 
}
void runAdvGBDTModel(const AdvGBDTParams& params, const DataParams& data_params) {
  std::cout << "============= Advanced GBDT Training =============" << std::endl;
  
  // Convert flat data to matrix format for advanced GBDT
  std::vector<std::vector<double>> X_train, X_test;
  to_matrix(data_params.X_train, data_params.rowLength, X_train);
  to_matrix(data_params.X_test, data_params.rowLength, X_test);
  
  // Feature names for importance calculation
  std::vector<std::string> feature_names = {
      "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "matrix_size_x", "matrix_size_y"
  };
  
  // Convert binning method enum
  ImprovedGBDT::BinningMethod bin_method = params.binMethod == AdvBinMethod::Quantile ? 
      ImprovedGBDT::BinningMethod::QUANTILE : ImprovedGBDT::BinningMethod::FREQUENCY;
  
  // Initialize model with optimized parameters
  ImprovedGBDT model(
    params.nEstimators,
    params.maxDepth,
    params.learningRate,
    params.useDart,            // Disable DART for stability
    params.dropoutRate,              // No dropout
    params.skipDropRate,              // No skip
    bin_method,
    params.numBins,
    params.minDataLeaf,
    1.0,              // L2 regularization - match LightGBM default
    0.8,              // Feature sampling ratio for randomization
    0,                 // Early stopping rounds
    params.numThreads
  );
  
  // Set OpenMP threads
  //omp_set_num_threads(params.numThreads);

  // Affichage des paramètres
  std::cout << "[AdvGBDT] Paramètres du modèle:" << std::endl;
  std::cout << "  n_estimators     = " << params.nEstimators << std::endl;
  std::cout << "  learning_rate    = " << params.learningRate << std::endl;
  std::cout << "  max_depth        = " << params.maxDepth << std::endl;
  std::cout << "  min_data_leaf    = " << params.minDataLeaf << std::endl;
  std::cout << "  num_bins         = " << params.numBins << std::endl;
  std::cout << "  use_dart         = " << params.useDart << std::endl;
  std::cout << "  dropout_rate     = " << params.dropoutRate << std::endl;
  std::cout << "  skip_drop_rate   = " << params.skipDropRate << std::endl;
  std::cout << "  binning_method   = " << (params.binMethod == AdvBinMethod::Quantile ? "QUANTILE" : "FREQUENCY") << std::endl;
  std::cout << "  num_threads      = " << params.numThreads << std::endl;
  
  // Train model
  auto train_start = std::chrono::high_resolution_clock::now();
  
  // Train without validation set for simplicity
  model.fit(X_train, data_params.y_train);
  
  auto train_end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration<double>(train_end - train_start).count();
  std::cout << "[AdvGBDT] Training time: " << train_time << " s" << std::endl;
  
  // Make predictions
  auto pred_start = std::chrono::high_resolution_clock::now();
  std::vector<double> predictions = model.predict(X_test);
  auto pred_end = std::chrono::high_resolution_clock::now();
  double pred_time = std::chrono::duration<double>(pred_end - pred_start).count();
  std::cout << "[AdvGBDT] Prediction time: " << pred_time << " s" << std::endl;
  
  // Calculate metrics
  double mse = 0.0, mae = 0.0;
  for (size_t i = 0; i < predictions.size(); ++i) {
      double diff = predictions[i] - data_params.y_test[i];
      mse += diff * diff;
      mae += std::abs(diff);
  }
  mse /= predictions.size();
  mae /= predictions.size();
  std::cout << "[AdvGBDT] MSE=" << mse << ", MAE=" << mae << std::endl;
  
  // Get feature importance
  auto importances = model.featureImportance();
  
  std::cout << "[AdvGBDT] Feature importance:" << std::endl;
  for (size_t i = 0; i < importances.size() && i < feature_names.size(); ++i) {
      std::cout << feature_names[i] << ": " << importances[i] * 100 << std::endl;
  }
  
  // Save model
  std::string model_file = "../saved_models/adv_gbdt_models/adv_gbdt_model.bin";
  model.saveModel(model_file);
  std::cout << "[AdvGBDT] Model saved to: " << model_file << std::endl;
  
  // Save results for comparison
  ModelResults results;
  results.model_name = "Advanced GBDT";
  results.mse_or_mae = mse;
  results.training_time = train_time;
  results.evaluation_time = pred_time;
  
  // Save parameters
  results.parameters["n_estimators"] = params.nEstimators;
  results.parameters["learning_rate"] = params.learningRate;
  results.parameters["max_depth"] = params.maxDepth;
  results.parameters["min_data_leaf"] = params.minDataLeaf;
  results.parameters["num_bins"] = params.numBins;
  results.parameters["use_dart"] = false;  // Disabled for now
  results.parameters["num_threads"] = params.numThreads;
  results.parameters["binning_method"] = static_cast<int>(params.binMethod);
  
  // Save feature importance
  for (size_t i = 0; i < importances.size() && i < feature_names.size(); ++i) {
      results.feature_importance[feature_names[i]] = importances[i];
  }
  
  ModelComparison::saveResults(results);
}