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
                 int whichLossFunc, bool useSplitHistogram, bool useOMP,
                 int numThreads)
    : numTrees(num_trees), maxDepth(max_depth),
      minSamplesSplit(min_samples_split),
      minImpurityDecrease(min_impurity_decrease),
      loss_function(std::move(loss_func)), Criteria(Criteria),
      whichLossFunc(whichLossFunc), useSplitHistogram(useSplitHistogram),
      useOMP(useOMP), numThreads(numThreads) {
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

// /**
//  * Train the Bagging ensemble
//  * @param data Flattened feature matrix (1D vector)
//  * @param rowLength Number of features per row/sample
//  * @param labels Target vector
//  * @param criteria Loss criteria (e.g., MSE or MAE)
//  */
// void Bagging::train(const std::vector<double> &data, int rowLength,
//                     const std::vector<double> &labels, int criteria) {
//   if (useOMP) {
//     std::vector<std::future<std::unique_ptr<DecisionTreeSingle>>> futures;
//     trees.clear();
//     trees.resize(numTrees);
//     omp_set_max_active_levels(2); //LIMIT the amount of nested parallelism
//     #pragma omp parallel for  \
//           num_threads(numThreads)       \
//           schedule(dynamic, 1)          \
//           default(none)                 \
//           shared(trees, data, labels)   \
//           firstprivate(rowLength, criteria)
//     for (int i = 0; i < numTrees; ++i) {

//         std::vector<double> sampled_data;
//         std::vector<double> sampled_labels;
//         bootstrapSample(data, rowLength, labels, sampled_data,
//         sampled_labels);

//         // Create and train a new DecisionTreeSingle
//         std::unique_ptr<DecisionTreeSingle> tree =
//         std::make_unique<DecisionTreeSingle>(maxDepth, minSamplesSplit,
//         minImpurityDecrease,
//                                                          criteria,
//                                                          useSplitHistogram,
//                                                          useOMP, numThreads);
//         tree->train(sampled_data, rowLength, sampled_labels, criteria);
//         trees[i] = std::move(tree);
//       };

//   } else {
//     for (int i = 0; i < numTrees; ++i) {
//       std::vector<double> sampled_data;
//       std::vector<double> sampled_labels;
//       bootstrapSample(data, rowLength, labels, sampled_data, sampled_labels);

//       // Create and train a new DecisionTreeSingle
//       std::unique_ptr<DecisionTreeSingle> tree =
//       std::make_unique<DecisionTreeSingle>(maxDepth, minSamplesSplit,
//                                                                                       minImpurityDecrease, criteria,
//                                                                                       useSplitHistogram, useOMP, numThreads);
//       tree->train(sampled_data, rowLength, sampled_labels, criteria);
//       trees.push_back(std::move(tree));
//     }
//   }
// }

void Bagging::train(const std::vector<double> &data, int rowLength,
                    const std::vector<double> &labels, int criteria) {
    // Detect whether MPI is running
  int mpiRank = 0, mpiSize = 1, mpiInit = 0;
#ifdef USE_MPI
  MPI_Initialized(&mpiInit);
  if (mpiInit) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  }
#endif

  auto isMyTree = [mpiRank, mpiSize](int t) { return t % mpiSize == mpiRank; };

  /* maintain a local vector even if we’ll later gather to rank-0 */
  std::vector<std::unique_ptr<DecisionTreeSingle>> localForest;

  if (useOMP)
    omp_set_max_active_levels(2); // keep two nesting layers
// OpenMP loop: build the trees that belong to this rank
#pragma omp parallel for schedule(dynamic, 1) default(none)                    \
    shared(data, labels, rowLength, criteria, numTrees, isMyTree, localForest) \
    firstprivate(maxDepth, minSamplesSplit, minImpurityDecrease,               \
                     useSplitHistogram, useOMP, numThreads)                    \
    num_threads(numThreads)
  for (int t = 0; t < numTrees; ++t) {
    if (!isMyTree(t))
      continue; // another rank will handle this

    std::vector<double> sampData, sampLabels;
    bootstrapSample(data, rowLength, labels, sampData, sampLabels);

    auto tree = std::make_unique<DecisionTreeSingle>(
        maxDepth, minSamplesSplit, minImpurityDecrease, criteria,
        useSplitHistogram, useOMP, numThreads);

    tree->train(sampData, rowLength, sampLabels, criteria);

#pragma omp critical
    localForest.push_back(std::move(tree));
  }
 //  If >1 rank, gather every tree on rank 0
 
#ifdef USE_MPI
  if (mpiSize > 1) {
   
    int localCount = static_cast<int>(localForest.size());
    std::vector<int> counts(mpiSize);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
               /*root=*/0, MPI_COMM_WORLD);
    if (mpiRank == 0) {
      trees.clear();
      trees.reserve(std::accumulate(counts.begin(), counts.end(), 0));

      for (auto &t : localForest)
        trees.push_back(std::move(t));

      for (int src = 1; src < mpiSize; ++src) {
        int bytes;
        MPI_Recv(&bytes, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<char> bigBuf(bytes);
        MPI_Recv(bigBuf.data(), bytes, MPI_BYTE, src, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        /* slice the big buffer back into individual trees  */
        int already = 0;
        for (int k = 0; k < counts[src]; ++k) {
          // Each tree is written back-to-back; we need its length.
          int len = *reinterpret_cast<int *>(&bigBuf[already]);
          already += sizeof(int);

          std::vector<char> oneTree(&bigBuf[already], &bigBuf[already + len]);
          trees.push_back(DecisionTreeSingle::deserializeFromBuffer(oneTree));
          already += len;
        }
      }
    } else { 
      // Pack all local trees into a single contiguous buffer
      std::vector<char> bigBuf;
      std::vector<int> offsets;
      offsets.reserve(localForest.size());

      for (auto &t : localForest) {
        std::vector<char> buf = t->serializeToBuffer();
        int len = static_cast<int>(buf.size());
        bigBuf.insert(bigBuf.end(), reinterpret_cast<char *>(&len),
                      reinterpret_cast<char *>(&len) + sizeof(int));
        bigBuf.insert(bigBuf.end(), buf.begin(), buf.end());
      }

      int bytes = static_cast<int>(bigBuf.size());
      MPI_Send(&bytes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(bigBuf.data(), bytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    }
  } else /* single rank build: keep localForest as member “trees” */
#endif
  {
    trees.swap(localForest);
  }
}

/**
 * Predict the target value for a single sample
 * @param sample Feature vector for the sample
 * @return Averaged prediction from all trees
 */
double Bagging::predict(const double *sample, int rowLength) const {
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
    const double *sample_ptr = &test_data[i * rowLength];
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
       << minImpurityDecrease << " " << Criteria << " " << whichLossFunc << " "
       << useSplitHistogram << " " << useOMP << " " << numThreads << "\n";

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
      Criteria >> whichLossFunc >> useSplitHistogram >> useOMP >> numThreads;

  // Réinitialiser et recharger les arbres
  trees.clear();
  trees.resize(numTrees);

  // Charger chaque arbre
  for (int i = 0; i < numTrees; ++i) {
    std::string tree_filename = filename + "_tree_" + std::to_string(i);
    trees[i] = std::make_unique<DecisionTreeSingle>(
        maxDepth, minSamplesSplit, minImpurityDecrease, Criteria,
        useSplitHistogram, numThreads);
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
  parameters["UseSplitHistogram"] = useSplitHistogram ? "true" : "false";
  parameters["UseOMP"] = useOMP ? "true" : "false";
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
  oss << "  - UseSplitHistogram: " << (useSplitHistogram ? "true" : "false")
      << "\n";
  oss << "  - UseOMP: " << (useOMP ? "true" : "false") << "\n";
  oss << " - Number of threads: " << numThreads << "\n";

  return oss.str();
}
