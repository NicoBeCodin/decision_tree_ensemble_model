#include "model_params.h"

bool getDecisionTreeParams(const ProgramOptions &options,
                           DecisionTreeParams &out_params) {
  // Create folder if non existent
  createDirectory("../saved_models/tree_models");

  if (options.use_custom_params && options.params.size() > 5) {
    out_params.criteria = std::stoi(options.params[0]);
    out_params.maxDepth = std::stoi(options.params[1]);
    out_params.minSamplesSplit = std::stoi(options.params[2]);
    out_params.minImpurityDecrease = std::stod(options.params[3]);
    out_params.useOMP = std::stoi(options.params[4]) != 0;
    out_params.numThreads = adjustNumThreads(std::stoi(
        options.params[5])); // This is to make sure it's a power of two
  } else if (options.load_request) {
    try {
      DecisionTreeSingle tmp_tree(0, 0, 0.0, 0); // Temporary
      tmp_tree.loadTree(options.path_model_filename);
      std::cout << "Model loaded successfully from "
                << options.path_model_filename << "\n";
      // Recover model parameters
      auto training_params = tmp_tree.getTrainingParameters();

      // Update parameter variables
      out_params.criteria = std::stoi(training_params["Criteria"]);
      out_params.maxDepth = std::stoi(training_params["MaxDepth"]);
      out_params.minSamplesSplit = std::stoi(training_params["MinLeafLarge"]);
      out_params.minImpurityDecrease = std::stod(training_params["MinError"]);
      out_params.useOMP = std::stoi(training_params["UseOMP"]) != 0;
      // Retrieve numThreads safely with a default value of 1, for
      // Retrocompatibility
      out_params.numThreads =
          (training_params.find("NumThreads") != training_params.end())
              ? std::stoi(training_params["NumThreads"])
              : 1;

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << tmp_tree.getTrainingParametersString() << "\n";

      return false; // Nothing else to do
    } catch (const std::exception &e) {
      std::cerr << "Failed to load tree model: " << e.what() << std::endl;
      return false;
    }
  } else {
    out_params.criteria = 0;
    out_params.maxDepth = 60;
    out_params.minSamplesSplit = 2;
    out_params.minImpurityDecrease = 1e-12;
    out_params.useOMP = false;
    out_params.numThreads = 1;
    std::cout << "Generation of default values : " << std::endl
              << "Default for splitting criteria (MSE)" << std::endl
              << "Default maximum depth = " << out_params.maxDepth << std::endl
              << "Default minimum sample split = " << out_params.minSamplesSplit << std::endl
              << "Default minimum impurity decrease = " << out_params.minImpurityDecrease << std::endl
              << "Default no useOMP = " << out_params.useOMP << std::endl
              << "Default number of threads : " << out_params.numThreads
              << " (OpenMP optimizations : off)" << std::endl;
  }
  return true;
}

bool getBaggingParams(const ProgramOptions &options,
                      BaggingParams &out_params) {
  DecisionTreeParams params;

  // Create folder if non existent
  createDirectory("../saved_models/bagging_models");

  if (options.use_custom_params && options.params.size() > 6) {
    out_params.criteria = std::stoi(options.params[0]);
    out_params.whichLossFunction = std::stoi(options.params[1]);
    out_params.numTrees = std::stoi(options.params[2]);
    out_params.maxDepth = std::stoi(options.params[3]);
    out_params.minSamplesSplit = std::stoi(options.params[4]);
    out_params.minImpurityDecrease = std::stod(options.params[5]);
    out_params.useOMP = std::stoi(options.params[6]) != 0;
    out_params.numThreads = std::stoi(options.params[7]);
    
  } else if (options.load_request) {
    try {
      Bagging tmp_bagging_model(0, 0, 0, 0.0, nullptr, 0, 0, 1); // Temp init
      tmp_bagging_model.load(options.path_model_filename);
      std::cout << "Model loaded successfully from "
                << options.path_model_filename << "\n";
      // Recover model parameters
      auto training_params = tmp_bagging_model.getTrainingParameters();

      // Update parameter variables
      out_params.numTrees = std::stoi(training_params["NumTrees"]);
      out_params.maxDepth = std::stoi(training_params["MaxDepth"]);
      out_params.minSamplesSplit = std::stoi(training_params["MinSamplesSplit"]);
      out_params.minImpurityDecrease = std::stod(training_params["MinImpurityDecrease"]);
      out_params.criteria = std::stoi(training_params["Criteria"]);
      out_params.whichLossFunction = std::stoi(training_params["WhichLossFunction"]);
      out_params.useOMP = std::stoi(training_params["UseOMP"]) != 0;
      out_params.numThreads = std::stoi(training_params["NumThreads"]);

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << tmp_bagging_model.getTrainingParametersString() << "\n";

      return false; // Nothing done but model loaded
    } catch (const std::runtime_error &e) {
      std::cerr << "Error loading the model: " << e.what() << "\n";
      return false;
    }
  } else {
    out_params.criteria = 0;
    out_params.whichLossFunction = 0;
    out_params.numTrees = 20;
    out_params.maxDepth = 60;
    out_params.minSamplesSplit = 2;
    out_params.minImpurityDecrease = 1e-6;
    out_params.useOMP = false;
    out_params.numThreads = 1;
    int mpiRank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
#endif
    if (mpiRank == 0) {
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)" << std::endl
                << "Default for comparing trees (MSE)" << std::endl
                << "Default number of trees to generate : " << out_params.numTrees << std::endl
                << "Default maximum depth = " << out_params.maxDepth << std::endl
                << "Default minimum sample split = " << out_params.minSamplesSplit << std::endl
                << "Default minimum impurity decrease = " << out_params.minImpurityDecrease << std::endl
                << "Default no useOMP = " << out_params.useOMP << std::endl
                << "Default amount of threads used : " << out_params.numThreads << std::endl;
    }
  }
  return true;
}

bool getBoostingParams(const ProgramOptions &options,
                       BoostingParams &out_params) {
  // Create boosting folder if new
  createDirectory("../saved_models/boosting_models");

  if (options.use_custom_params && options.params.size() > 6) {
    out_params.criteria = std::stoi(options.params[0]);
    out_params.whichLossFunction = std::stoi(options.params[1]);
    out_params.nEstimators = std::stoi(options.params[2]);
    out_params.maxDepth = std::stoi(options.params[3]);
    out_params.minSamplesSplit = std::stoi(options.params[4]);
    out_params.minImpurityDecrease = std::stod(options.params[5]);
    out_params.learningRate = std::stod(options.params[6]);
    out_params.useOMP = std::stoi(options.params[7]) != 0;
    out_params.numThreads = std::stoi(options.params[8]);
  } else if (options.load_request) {
    try {
      Boosting tmp_boosting_model(0, 0.0, nullptr, 0, 0, 0.0, 0,
                                  0); // temporary creation
      tmp_boosting_model.load(options.path_model_filename);
      std::cout << "Model loaded successfully from "
                << options.path_model_filename << "\n";
      // Recover model parameters
      auto training_params = tmp_boosting_model.getTrainingParameters();

      // Update parameter variables
      out_params.nEstimators = std::stoi(training_params["NumEstimators"]);
      out_params.learningRate = std::stod(training_params["LearningRate"]);
      out_params.maxDepth = std::stoi(training_params["MaxDepth"]);
      out_params.minSamplesSplit = std::stoi(training_params["MinSamplesSplit"]);
      out_params.minImpurityDecrease = std::stod(training_params["MinImpurityDecrease"]);
      double initial_prediction = std::stod(training_params["InitialPrediction"]);
      out_params.criteria = std::stoi(training_params["Criteria"]);
      out_params.whichLossFunction = std::stoi(training_params["WhichLossFunction"]);
      out_params.useOMP = std::stoi(training_params["UseOMP"]) != 0;
      out_params.numThreads = std::stoi(training_params["NumThreads"]);

      // Display tree parameters
      std::cout << "Parameters loaded from the model file:\n";
      std::cout << tmp_boosting_model.getTrainingParametersString() << "\n";

      return false; // Nothing done but model loaded
    } catch (const std::runtime_error &e) {
      std::cerr << "Error loading the model: " << e.what() << "\n";
      return false;
    }
  } else {
    out_params.criteria = 0;
    out_params.whichLossFunction = 0;
    out_params.nEstimators = 75;
    out_params.maxDepth = 15;
    out_params.minSamplesSplit = 3;
    out_params.minImpurityDecrease = 1e-5;
    out_params.learningRate = 0.07;
    out_params.useOMP = false;
    out_params.numThreads = 1;
    std::cout << "Generation of default values : " << std::endl
              << "Default for splitting criteria (MSE)" << std::endl
              << "Default for comparing trees (MSE)" << std::endl
              << "Default number of estimators : " << out_params.nEstimators << std::endl
              << "Default maximum depth = " << out_params.maxDepth << std::endl
              << "Default minimum sample split = " << out_params.minSamplesSplit << std::endl
              << "Default minimum impurity decrease = " << out_params.minImpurityDecrease << std::endl
              << "Default learning rate = " << out_params.learningRate << std::endl
              << "Default no useOMP = " << out_params.useOMP << std::endl
              << "Default amount of threads used : " << out_params.numThreads << std::endl;
  }
  return true;
}

bool getLightGBMParams(const ProgramOptions &options,
                       LightGBMParams &out_params) {

  createDirectory("../saved_models/lightgbm_models");

  if (options.use_custom_params && options.params.size() >= 6) {
    out_params.nEstimators = std::stoi(options.params[0]);
    out_params.learningRate = std::stod(options.params[1]);
    out_params.maxDepth = std::stoi(options.params[2]);
    out_params.numLeaves = std::stoi(options.params[3]);
    out_params.subsample = std::stod(options.params[4]);
    out_params.colsampleBytree = std::stod(options.params[5]);
  } else if (options.load_request) {

    std::cout << "Loading existing LightGBM 模型："
              << options.path_model_filename << std::endl;
    return false;
  } else {

    out_params.nEstimators = 100;
    out_params.learningRate = 0.1;
    out_params.maxDepth = -1;
    out_params.numLeaves = 31;
    out_params.subsample = 1.0;
    out_params.colsampleBytree = 1.0;
    std::cout << "Using default LightGBM parameters." << std::endl;
  }
  return true;
}

bool getAdvGBDTParams(const ProgramOptions& options, AdvGBDTParams& out_params) {
  createDirectory("../saved_models/adv_gbdt_models");
  
  // Debug output
  std::cout << "use_custom_params: " << (options.use_custom_params ? "true" : "false") << std::endl;
  std::cout << "params.size(): " << options.params.size() << std::endl;
  
  if (options.use_custom_params && options.params.size() >= 10) {
      std::cout << "Using custom parameters:" << std::endl;
      
      try {
          out_params.nEstimators    = std::stoi(options.params[0]);
          out_params.learningRate   = std::stod(options.params[1]);
          out_params.maxDepth       = std::stoi(options.params[2]);
          out_params.minDataLeaf    = std::stoul(options.params[3]);
          out_params.numBins        = std::stoi(options.params[4]);
          out_params.useDart        = std::stoi(options.params[5]) != 0;
          out_params.dropoutRate    = std::stod(options.params[6]);
          out_params.skipDropRate   = std::stod(options.params[7]);
          out_params.numThreads     = std::stoi(options.params[8]);
          out_params.binMethod      = (std::stoi(options.params[9]) == 0) ? AdvBinMethod::Quantile : AdvBinMethod::Frequency;
      } catch (const std::exception& e) {
          std::cerr << "Error parsing parameters: " << e.what() << std::endl;
          std::cerr << "Using LightGBM-like defaults instead." << std::endl;
          
          // Parameters similar to LightGBM's defaults which perform well
          out_params = {
              100,    // nEstimators - like LightGBM default
              0.1,    // learningRate - standard value that works well
              6,      // maxDepth - shallow trees often work better
              20,     // minDataLeaf - increased for stability
              255,    // numBins - standard LightGBM value
              false,  // useDart - disabled for stability
              0.0,    // dropoutRate - not used
              0.0,    // skipDropRate - not used
              8,      // numThreads
              AdvBinMethod::Frequency  // binMethod - often better for regression
          };
      }
  } else {
      std::cout << "Using LightGBM-like default parameters" << std::endl;
      
      // Parameters similar to LightGBM's defaults which perform well
      out_params = {
          200,    // nEstimators - like LightGBM default
          0.1,    // learningRate - standard value that works well
          6,      // maxDepth - shallow trees often work better
          20,     // minDataLeaf - increased for stability
          255,    // numBins - standard LightGBM value
          true,  // useDart - disabled for stability
          0.5,    // dropoutRate - not used
          0.3,    // skipDropRate - not used
          8,      // numThreads
          AdvBinMethod::Frequency  // binMethod - often better for regression
      };
  }
  
  // Display parameters
  std::cout << "Parameters for AdvGBDT:" << std::endl
            << "- Number of estimators: " << out_params.nEstimators << std::endl
            << "- Learning rate: " << out_params.learningRate << std::endl
            << "- Max depth: " << out_params.maxDepth << std::endl
            << "- Min samples per leaf: " << out_params.minDataLeaf << std::endl
            << "- Number of bins: " << out_params.numBins << std::endl
            << "- Using DART: " << (out_params.useDart ? "Yes" : "No") << std::endl;
  
  if (out_params.useDart) {
      std::cout << "- Dropout rate: " << out_params.dropoutRate << std::endl
                << "- Skip dropout rate: " << out_params.skipDropRate << std::endl;
  }
  
  std::cout << "- Number of threads: " << out_params.numThreads << std::endl
            << "- Binning method: " << (out_params.binMethod == AdvBinMethod::Quantile ? "Quantile" : "Frequency") << std::endl;
  
  return true;
}