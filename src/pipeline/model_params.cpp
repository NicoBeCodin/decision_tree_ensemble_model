#include "model_params.h"
  
bool getDecisionTreeParams(const ProgramOptions& options, DecisionTreeParams& out_params) {
    // Create folder if non existent
    createDirectory("../saved_models/tree_models");
  
    if (options.use_custom_params && options.params.size() > 5) {
      out_params.criteria = std::stoi(options.params[0]);
      out_params.maxDepth = std::stoi(options.params[1]);
      out_params.minSamplesSplit = std::stoi(options.params[2]);
      out_params.minImpurityDecrease = std::stod(options.params[3]);
      out_params.useSplitHistogram = std::stoi(options.params[4]) != 0;
      out_params.numThreads = adjustNumThreads(std::stoi(options.params[5])); // This is to make sure it's a power of two
    } else if (options.load_request) {
      try {
        DecisionTreeSingle tmp_tree(0, 0, 0.0, 0); // Temporary
        tmp_tree.loadTree(options.path_model_filename);
        std::cout << "Model loaded successfully from " << options.path_model_filename << "\n";
        // Recover model parameters
        auto training_params = tmp_tree.getTrainingParameters();
  
        // Update parameter variables
        out_params.criteria = std::stoi(training_params["Criteria"]);
        out_params.maxDepth = std::stoi(training_params["MaxDepth"]);
        out_params.minSamplesSplit = std::stoi(training_params["MinLeafLarge"]);
        out_params.minImpurityDecrease = std::stod(training_params["MinError"]);
        out_params.useSplitHistogram = std::stoi(training_params["UseSplitHistogram"]) != 0;
        // Retrieve numThreads safely with a default value of 1, for
        // Retrocompatibility
        out_params.numThreads = (training_params.find("NumThreads") != training_params.end()) ? std::stoi(training_params["NumThreads"]) : 1;
  
        // Display tree parameters
        std::cout << "Parameters loaded from the model file:\n";
        std::cout << tmp_tree.getTrainingParametersString() << "\n";
        
        return false; // Nothing else to do
      } catch (const std::exception& e) {
        std::cerr << "Failed to load tree model: " << e.what() << std::endl;
        return false;
      }
    } else {
      out_params.criteria = 0;
      out_params.maxDepth = 60;
      out_params.minSamplesSplit = 2;
      out_params.minImpurityDecrease = 1e-12;
      out_params.useSplitHistogram = false;
      out_params.numThreads = 1;
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)" << std::endl
                << "Default maximum depth = " << out_params.maxDepth << std::endl
                << "Default minimum sample split = " << out_params.minSamplesSplit << std::endl
                << "Default minimum impurity decrease = " << out_params.minImpurityDecrease << std::endl
                << "Default no useSplitHistogram = " << out_params.minImpurityDecrease << std::endl
                << "Default number of threads : " << out_params.numThreads << " (OpenMP optimizations : off)" << std::endl;
    }
    return true;
}

bool getBaggingParams(const ProgramOptions& options, BaggingParams& out_params) {
    DecisionTreeParams params;
  
    // Create folder if non existent
    createDirectory("../saved_models/bagging_models");
  
    if (options.use_custom_params && options.params.size() > 5) {
      out_params.criteria = std::stoi(options.params[0]);
      out_params.whichLossFunction = std::stoi(options.params[1]);
      out_params.numTrees = std::stoi(options.params[2]);
      out_params.maxDepth = std::stoi(options.params[3]);
      out_params.minSamplesSplit = std::stoi(options.params[4]);
      out_params.minImpurityDecrease = std::stod(options.params[5]);
      out_params.useSplitHistogram = std::stoi(options.params[6]) != 0;
      out_params.numThreads = std::stoi(options.params[7]);
    } else if (options.load_request) {
      try {
        Bagging tmp_bagging_model(0, 0, 0, 0.0, nullptr, 0, 0, 1); // Temp init
        tmp_bagging_model.load(options.path_model_filename);
        std::cout << "Model loaded successfully from " << options.path_model_filename << "\n";
        // Recover model parameters
        auto training_params = tmp_bagging_model.getTrainingParameters();
  
        // Update parameter variables
        out_params.numTrees = std::stoi(training_params["NumTrees"]);
        out_params.maxDepth = std::stoi(training_params["MaxDepth"]);
        out_params.minSamplesSplit = std::stoi(training_params["MinSamplesSplit"]);
        out_params.minImpurityDecrease = std::stod(training_params["MinImpurityDecrease"]);
        out_params.criteria = std::stoi(training_params["Criteria"]);
        out_params.whichLossFunction = std::stoi(training_params["WhichLossFunction"]);
        out_params.useSplitHistogram = std::stoi(training_params["UseSplitHistogram"]) != 0;
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
      out_params.useSplitHistogram = false;
      out_params.numThreads = 1;
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)" << std::endl
                << "Default for comparing trees (MSE)" << std::endl
                << "Default number of trees to generate : " << out_params.numTrees << std::endl
                << "Default maximum depth = " << out_params.maxDepth << std::endl
                << "Default minimum sample split = " << out_params.minSamplesSplit << std::endl
                << "Default minimum impurity decrease = " << out_params.minImpurityDecrease << std::endl
                << "Default no useSplitHistogram = " << out_params.minImpurityDecrease << std::endl
                << "Default amount of threads used : " << out_params.numThreads << std::endl;
    }
    return true;
}

bool getBoostingParams(const ProgramOptions& options, BoostingParams& out_params) {
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
      out_params.useSplitHistogram = std::stoi(options.params[7]) != 0;
      out_params.numThreads = std::stoi(options.params[8]);
    } else if (options.load_request) {
      try {
        Boosting tmp_boosting_model(0, 0.0, nullptr, 0, 0, 0.0, 0, 0); // temporary creation
        tmp_boosting_model.load(options.path_model_filename);
        std::cout << "Model loaded successfully from " << options.path_model_filename << "\n";
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
        out_params.useSplitHistogram = std::stoi(training_params["UseSplitHistogram"]) != 0;
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
      out_params.useSplitHistogram = true;
      out_params.numThreads = 1;
      std::cout << "Generation of default values : " << std::endl
                << "Default for splitting criteria (MSE)" << std::endl
                << "Default for comparing trees (MSE)" << std::endl
                << "Default number of estimators : " << out_params.nEstimators << std::endl
                << "Default maximum depth = " << out_params.maxDepth << std::endl
                << "Default minimum sample split = " << out_params.minSamplesSplit << std::endl
                << "Default minimum impurity decrease = " << out_params.minImpurityDecrease << std::endl
                << "Default learning rate = " << out_params.learningRate << std::endl
                << "Default useSplitHistogram = " << out_params.minImpurityDecrease << std::endl
                << "Default amount of threads used : " << out_params.numThreads << std::endl;
    }
    return true;
}

bool getLightGBMParams(const ProgramOptions& options, LightGBMParams& out_params) {
  
  createDirectory("../saved_models/lightgbm_models");

  if (options.use_custom_params && options.params.size() >= 6) {
      out_params.nEstimators   = std::stoi(options.params[0]);
      out_params.learningRate  = std::stod(options.params[1]);
      out_params.maxDepth      = std::stoi(options.params[2]);
      out_params.numLeaves     = std::stoi(options.params[3]);
      out_params.subsample     = std::stod(options.params[4]);
      out_params.colsampleBytree = std::stod(options.params[5]);
  }
  else if (options.load_request) {
      
      std::cout << "Loading existing LightGBM 模型：" 
                << options.path_model_filename << std::endl;
      return false;
  }
  else {
      
      out_params.nEstimators   = 100;
      out_params.learningRate  = 0.1;
      out_params.maxDepth      = -1;
      out_params.numLeaves     = 31;
      out_params.subsample     = 1.0;
      out_params.colsampleBytree = 1.0;
      std::cout << "Using default LightGBM parameters." << std::endl;
  }
  return true;
}

bool getAdvGBDTParams(const ProgramOptions& o, AdvGBDTParams& p) {
  createDirectory("../saved_models/adv_gbdt_models");
  
  // Débogage
  std::cout << "use_custom_params: " << (o.use_custom_params ? "true" : "false") << std::endl;
  std::cout << "params.size(): " << o.params.size() << std::endl;
  
  if (o.use_custom_params && o.params.size() >= 10) {
      std::cout << "Utilisation des paramètres personnalisés:" << std::endl;
      
      p.nEstimators    = std::stoi(o.params[0]);
      p.learningRate   = std::stod(o.params[1]);
      p.maxDepth       = std::stoi(o.params[2]);
      p.minDataLeaf    = std::stoul(o.params[3]);
      p.numBins        = std::stoi(o.params[4]);
      p.useDart        = std::stoi(o.params[5]) != 0;
      p.dropoutRate    = std::stod(o.params[6]);
      p.skipDropRate   = std::stod(o.params[7]);
      p.numThreads     = std::stoi(o.params[8]);
      p.binMethod      = (std::stoi(o.params[9]) == 0) ? AdvBinMethod::Quantile : AdvBinMethod::Frequency;
      
      std::cout << "Paramètres personnalisés chargés avec succès." << std::endl;
  } else {
      std::cout << "Utilisation des valeurs par défaut car use_custom_params=" 
                << (o.use_custom_params ? "true" : "false") 
                << " et params.size()=" << o.params.size() << std::endl;
      
                p = {200, 0.01, 50, 1, 1024, 1, 0.5, 0.3, 8, AdvBinMethod::Frequency};
  }
  
  // Affichage des paramètres qui seront utilisés
  std::cout << "Paramètres effectifs pour AdvGBDT:" << std::endl
            << "- Nombre d'estimateurs: " << p.nEstimators << std::endl
            << "- Taux d'apprentissage: " << p.learningRate << std::endl
            << "- Profondeur maximale: " << p.maxDepth << std::endl
            << "- Minimum d'échantillons par feuille: " << p.minDataLeaf << std::endl
            << "- Nombre de bins: " << p.numBins << std::endl
            << "- Utilisation de DART: " << (p.useDart ? "Oui" : "Non") << std::endl
            << "- Taux de dropout: " << p.dropoutRate << std::endl
            << "- Taux de skip: " << p.skipDropRate << std::endl
            << "- Nombre de threads: " << p.numThreads << std::endl
            << "- Méthode de binning: " << (p.binMethod == AdvBinMethod::Quantile ? "Quantile" : "Frequency") << std::endl;
  
  return true;
}