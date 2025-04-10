#include "../functions/math/math_functions.h"
#include "../pipeline/model_params.h"
#include "../pipeline/run_models.h"
#include "../pipeline/data_split.h"
#include <chrono>
#include <iomanip>
#include <memory>

int main(int argc, char *argv[]) {

  ProgramOptions programOptions = parseCommandLineArguments(argc, argv);

  DataParams data_params;

  splitDataset(data_params);
  
  switch (programOptions.choice) {
    case 1: {
      DecisionTreeParams treeParams;
      if (!getDecisionTreeParams(programOptions, treeParams)) {
        return -1; // Successful or unsuccessful loading
      }
      runSingleDecisionTreeModel(treeParams, data_params); break;
    }
    case 2: {
      BaggingParams baggingParams;
      if (!getBaggingParams(programOptions, baggingParams)) {
        return -1; // Successful or unsuccessful loading
      }
      runBaggingModel(baggingParams, data_params); break;
    }
    case 3: {
      BoostingParams boostingParams;
      if (!getBoostingParams(programOptions, boostingParams)) {
        return -1; // Successful or unsuccessful loading
      }
      runBoostingModel(boostingParams, data_params); break;
    }
    default: std::cerr << "Invalid choice! Please choose 1, 2, 3 or 4" << std::endl; return -1;
  }
  return 0;
}