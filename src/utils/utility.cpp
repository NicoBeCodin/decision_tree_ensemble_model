#include "utility.h"


// For the single decision tree multithreading
int adjustNumThreads(int numThreads) {
  if (numThreads <= 0)
    return 1;
  if ((numThreads & (numThreads - 1)) == 0)
    return numThreads;

  int power = 1;
  while (power * 2 <= numThreads)
    power *= 2;

  return power;
}

// input function to set parameters with defaults
template <typename T>
T getInputWithDefault(const std::string &prompt, T defaultValue) {
  std::cout << prompt << " (Default: " << defaultValue << "): ";
  std::string input;
  std::getline(std::cin, input); // Read user input as string

  // If empty return default
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

ProgramOptions parseCommandLineArguments(int argc, char *argv[]) {
  ProgramOptions options;

  if (argc > 1) {
    // First positional argument is the model choice
    options.choice = std::stoi(argv[1]);

    int i = 2;
    while (i < argc) {
      std::string arg = argv[i];
      if (arg == "-p" && i + 1 < argc) {
        options.use_custom_params = true;
        i += 2;
      } else if (arg == "-l" && i + 1 < argc) {
        options.load_request = true;
        options.path_model_filename = argv[i + 1];
        i += 2;
      } else if (arg.rfind("--", 0) == 0) {
        // CLI flag in form --key=value or --key value
        auto eq = arg.find('=');
        if (eq != std::string::npos) {
          std::string key = arg.substr(2, eq - 2);
          std::string val = arg.substr(eq + 1);
          options.flags[key] = val;
        } else if (i + 1 < argc) {
          std::string key = arg.substr(2);
          std::string val = argv[i + 1];
          options.flags[key] = val;
          i++;
        }
        i++;
      } else {
        // Any other positional parameters
        options.params.push_back(arg);
        i++;
      }
    }
  } else {
    // Interactive prompt if no arguments passed
    std::cout << "Choose the method you want to use:\n"
              << "1: Simple Decision Tree\n"
              << "2: Bagging\n"
              << "3: Boosting\n"
              << "4: LightGBM\n"
              << "5: Advanced GBDT\n";
    std::cin >> options.choice;
  }
  return options;
}

void createDirectory(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
    std::cout << "Directory created: " << path << std::endl;
  }
}
