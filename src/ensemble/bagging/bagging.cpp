#include "bagging.h"

/**
 * Constructor for Bagging
 * @param num_trees Number of trees in the Bagging ensemble
 * @param max_depth Maximum depth of each tree
 * @param min_samples_split Minimum number of samples required to split a node
 * @param min_impurity_decrease Minimum impurity decrease required for a split
 * @param loss_function
 */
Bagging::Bagging(int num_trees, int max_depth, int min_samples_split,
                 double min_impurity_decrease,
                 std::unique_ptr<LossFunction> loss_func, int Criteria,
                 int whichLossFunc, int numThreads)
    : numTrees(num_trees), maxDepth(max_depth),
      minSamplesSplit(min_samples_split),
      minImpurityDecrease(min_impurity_decrease),
      loss_function(std::move(loss_func)), Criteria(Criteria),
      whichLossFunc(whichLossFunc), numThreads(numThreads) {
  trees.reserve(numTrees); // Reserve space for the trees
}

/**
 * Generate a bootstrap sample from the dataset
 * @param data Flattened feature matrix (1D vector)
 * @param rowLength Number of features per row/sample
 * @param labels Original dataset's target vector
 * @param sampled_data Output parameter for the sampled feature matrix
 * (flattened)
 * @param sampled_labels Output parameter for the sampled target vector
 */
void Bagging::bootstrapSample(const std::vector<double> &data, int rowLength,
                              const std::vector<double> &labels,
                              std::vector<double> &sampled_data,
                              std::vector<double> &sampled_labels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, labels.size() - 1);

  size_t n_samples = labels.size();
  sampled_data.reserve(n_samples * rowLength);
  sampled_labels.reserve(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    int idx = dis(gen);
    sampled_labels.push_back(labels[idx]);
    sampled_data.insert(sampled_data.end(), data.begin() + idx * rowLength,
                        data.begin() + (idx + 1) * rowLength);
  }
}

/**
 * Train the Bagging ensemble
 * @param data Flattened feature matrix (1D vector)
 * @param rowLength Number of features per row/sample
 * @param labels Target vector
 * @param criteria Loss criteria (e.g., MSE or MAE)
 */
void Bagging::train(const std::vector<double> &data, int rowLength,
                    const std::vector<double> &labels, int criteria) {
  if (numThreads == 1) {
    for (int i = 0; i < numTrees; ++i) {
      std::vector<double> sampled_data;
      std::vector<double> sampled_labels;
      bootstrapSample(data, rowLength, labels, sampled_data, sampled_labels);

      // Create and train a new DecisionTreeSingle
      auto tree = std::make_unique<DecisionTreeSingle>(
          maxDepth, minSamplesSplit, minImpurityDecrease, criteria);
      tree->train(sampled_data, rowLength, sampled_labels, criteria);
      trees.push_back(std::move(tree));
    }
  } else if (numThreads > 1) {

    std::vector<std::future<std::unique_ptr<DecisionTreeSingle>>> futures;

    for (int i = 0; i < numTrees; ++i) {
      futures.push_back(std::async(std::launch::async, [this, &data, rowLength,
                                                        &labels, criteria]() {
        std::vector<double> sampled_data;
        std::vector<double> sampled_labels;
        bootstrapSample(data, rowLength, labels, sampled_data, sampled_labels);

        // Create and train a new DecisionTreeSingle
        auto tree = std::make_unique<DecisionTreeSingle>(
            maxDepth, minSamplesSplit, minImpurityDecrease, criteria, numThreads);
        tree->train(sampled_data, rowLength, sampled_labels, criteria);
        return tree; // Return trained tree
      }));

      // Limit concurrent threads to `numThreads`
      if (futures.size() >= numThreads) {
        for (auto &future : futures) {
          trees.push_back(
              std::move(future.get())); // Retrieve result and store in `trees`
        }
        futures.clear(); // Clear futures vector to free threads for next batch
      }
    }

    // Ensure all remaining trees are retrieved
    for (auto &future : futures) {
      trees.push_back(std::move(future.get()));
    }
  } else {
    std::cout << "Invalid thread number" << std::endl;
  }
}

/**
 * Predict the target value for a single sample
 * @param sample Feature vector for the sample
 * @return Averaged prediction from all trees
 */
double Bagging::predict(const double* sample, int rowLength) const {
  double sum = 0.0;
  for (const auto &tree : trees) {
    sum += tree->predict(sample, rowLength);
  }
  return sum / trees.size(); // Return the average prediction
}

/**
 * Evaluate the Bagging model on a test dataset
 * @param test_data Flattened feature matrix (1D vector) for the test set
 * @param rowLength Number of features per row/sample
 * @param test_labels Target vector of the test set
 * @return Computed loss depending on the specified loss function
 */
double Bagging::evaluate(const std::vector<double> &test_data, int rowLength,
                         const std::vector<double> &test_labels) const {

  std::vector<double> predictions;
    size_t n_samples = test_labels.size();
    predictions.reserve(n_samples); // optimisation : éviter reallocations

    for (size_t i = 0; i < n_samples; ++i) {
        const double* sample_ptr = &test_data[i * rowLength];
        predictions.push_back(predict(sample_ptr, rowLength));
    }
    return loss_function->computeLoss(test_labels, predictions);
}

/**
 * Save the Bagging model to a file
 * @param filename The filename to save the model to
 */
void Bagging::save(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Sauvegarder tous les paramètres du modèle
  file << numTrees << " " << maxDepth << " " << minSamplesSplit << " "
       << minImpurityDecrease << " " << Criteria << " " << whichLossFunc
       << "\n";

  // Sauvegarder chaque arbre avec un nom unique
  for (size_t i = 0; i < trees.size(); ++i) {
    std::string tree_filename = filename + "_tree_" + std::to_string(i);
    trees[i]->saveTree(tree_filename);
  }

  file.close();
}

/**
 * Load the Bagging model from a file
 * @param filename The filename to load the model from
 */
void Bagging::load(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  // Charger tous les paramètres du modèle
  file >> numTrees >> maxDepth >> minSamplesSplit >> minImpurityDecrease >>
      Criteria >> whichLossFunc ; //I don't add numThreads as this is a training parameters and not a tree parameter

  // Réinitialiser et recharger les arbres
  trees.clear();
  trees.resize(numTrees);

  // Charger chaque arbre
  for (int i = 0; i < numTrees; ++i) {
    std::string tree_filename = filename + "_tree_" + std::to_string(i);
    trees[i] = std::make_unique<DecisionTreeSingle>(maxDepth, minSamplesSplit,
                                                    minImpurityDecrease);
    trees[i]->loadTree(tree_filename);
  }

  file.close();
}

// Retourne les paramètres d'entraînement sous forme de dictionnaire
// (clé-valeur)
std::map<std::string, std::string> Bagging::getTrainingParameters() const {
  std::map<std::string, std::string> parameters;
  parameters["NumTrees"] = std::to_string(numTrees);
  parameters["MaxDepth"] = std::to_string(maxDepth);
  parameters["MinSamplesSplit"] = std::to_string(minSamplesSplit);
  parameters["MinImpurityDecrease"] = std::to_string(minImpurityDecrease);
  parameters["Criteria"] = std::to_string(Criteria);
  parameters["WhichLossFunction"] = std::to_string(whichLossFunc);
  parameters["NumThreads"] = std::to_string(numThreads);
  return parameters;
}

// Retourne les paramètres d'entraînement sous forme d'une chaîne de caractères
// lisible
std::string Bagging::getTrainingParametersString() const {
  std::ostringstream oss;
  oss << "Training Parameters:\n";
  oss << "  - Number of Trees: " << numTrees << "\n";
  oss << "  - Max Depth: " << maxDepth << "\n";
  oss << "  - Min Samples Split: " << minSamplesSplit << "\n";
  oss << "  - Min Impurity Decrease: " << minImpurityDecrease << "\n";
  oss << "  - Criteria: " << (Criteria == 0 ? "MSE" : "MAE") << "\n";
  oss << "  - Loss Function: "
      << (whichLossFunc == 0 ? "Least Squares Loss" : "Mean Absolute Loss")
      << "\n";
  oss << " - Number of threads: " << numThreads << "\n";

  return oss.str();
}
