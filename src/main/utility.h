#ifndef UTILITY
#define UTILITY

#include "../functions/feature/feature_importance.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>

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

void createDirectory(const std::string& path);
template <typename ModelType>
void saveModel(ModelType& model);
#endif
