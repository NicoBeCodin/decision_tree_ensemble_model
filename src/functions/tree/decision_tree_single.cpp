#include "decision_tree_single.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <tuple>

struct BestSplit {
  double imp;
  int feat;
  double thr;
};
// We define a struct to optimize reduction with OMP
#pragma omp declare reduction(maxBest                                          \
:BestSplit : omp_out = (omp_in.imp > omp_out.imp ? omp_in : omp_out))          \
    initializer(omp_priv = {0.0, -1, 0.0})

// Constructor
DecisionTreeSingle::DecisionTreeSingle(int MaxDepth, int MinLeafLarge,
                                       double MinError, int Criteria,
                                       bool useOMP, int numThreads)
    : MaxDepth(MaxDepth), MinLeafLarge(MinLeafLarge), MinError(MinError),
      Criteria(Criteria), useOMP(useOMP),
      Root(nullptr), numThreads(numThreads) {
  getMaxSplitDepth();
}

// Training function
void DecisionTreeSingle::train(const std::vector<double> &Data, int rowLength,
                               const std::vector<double> &Labels,
                               int criteria) {
  Root = std::make_unique<Tree>();
  // The importance of the features can be properly weighted
  RootSamples = Labels.size();
  RootMSE = (criteria == 0)
                ? Math::calculateMSE(Labels)
                : Math::calculateMAE(Labels, Math::calculateMedian(Labels));
  std::vector<int> Indices(Labels.size());
  std::iota(Indices.begin(), Indices.end(), 0);

  if (useOMP) {
#pragma omp parallel num_threads(numThreads)
#pragma omp single
    {
      if (criteria == 0) {
        splitNode(Root.get(), Data, rowLength, Labels, Indices, 0);
      } else if (criteria == 1) {
        splitNodeMAE(Root.get(), Data, rowLength, Labels, Indices, 0);
      }
    }
  } else {
    // MSE criterion
    if (criteria == 0) {
      splitNode(Root.get(), Data, rowLength, Labels, Indices, 0);
    }
    // Will use MAE criterion
    else if (criteria == 1) {
      splitNodeMAE(Root.get(), Data, rowLength, Labels, Indices, 0);
    }
  }
}

void DecisionTreeSingle::evaluate(const std::vector<double> &X_test,
                                  const int rowLength,
                                  const std::vector<double> &y_test,
                                  double &mse_value, double &mae_value) {
  size_t test_size = y_test.size();
  std::vector<double> y_pred;
  y_pred.reserve(test_size);

  for (size_t i = 0; i < test_size; ++i) {
    const double *sample_ptr = &X_test[i * rowLength];
    y_pred.push_back(predict(sample_ptr, rowLength));
  }
  mse_value = Math::computeLossMSE(y_test, y_pred);
  mae_value = Math::computeLossMAE(y_test, y_pred);
}

// Split node function (using MSE)
void DecisionTreeSingle::splitNode(Tree *Node, const std::vector<double> &Data,
                                   int rowLength,
                                   const std::vector<double> &Labels,
                                   const std::vector<int> &Indices, int Depth) {
  // Compute node metrics
  Node->NodeMetric = Math::calculateMSEWithIndices(Labels, Indices);
  Node->NodeSamples = Indices.size();

  // Stopping conditions
  if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) ||
      Node->NodeMetric < MinError) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices);
    return;
  }

  int BestFeature;
  double BestThreshold;
  double BestImpurityDecrease;

  if (numThreads < 1) {
    throw std::invalid_argument("numThreads must be >= 1");
  }

  // Find the best split
  if (useOMP) {
    std::tie(BestFeature, BestThreshold, BestImpurityDecrease) = 
    findBestSplitOMP(Data, rowLength, Labels, Indices, Node->NodeMetric);
  } else {
    std::tie(BestFeature, BestThreshold, BestImpurityDecrease) = 
    findBestSplit(Data, rowLength, Labels, Indices, Node->NodeMetric);
  }

  if (BestFeature == -1) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices);
    return;
  }

  Node->FeatureIndex = BestFeature;
  Node->MaxValue = BestThreshold;

  // Split data
  std::vector<int> LeftIndices, RightIndices;
  LeftIndices.reserve(Indices.size() / 2); // reduces memory reallocations
  RightIndices.reserve(Indices.size() / 2);

  for (int Idx : Indices) {
    if (Data[Idx * rowLength + BestFeature] <= BestThreshold) {
      LeftIndices.push_back(Idx);
    } else {
      RightIndices.push_back(Idx);
    }
  }

  Node->Left = std::make_unique<Tree>();
  Node->Right = std::make_unique<Tree>();

  // Parallelism goes here

  if (LeftIndices.empty() || RightIndices.empty()) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMeanWithIndices(
        Labels, Indices); // Use mean instead of median
    return;
  }

  if (useOMP && Depth < maxSplitDepth && !omp_in_final()) {
    // Grab the child pointers once ─ avoids capturing "Node" in each task
    auto *leftChild = Node->Left.get();
    auto *rightChild = Node->Right.get();

#pragma omp task shared(Data, Labels)                                          \
    firstprivate(leftChild, rowLength, LeftIndices, Depth)
    splitNode(leftChild, Data, rowLength, Labels, LeftIndices, Depth + 1);

#pragma omp task shared(Data, Labels)                                          \
    firstprivate(rightChild, rowLength, RightIndices, Depth)
    splitNode(rightChild, Data, rowLength, Labels, RightIndices, Depth + 1);

#pragma omp taskwait

  } else {
    // Perform normal sequential recursion after depth 2
    splitNode(Node->Left.get(), Data, rowLength, Labels, LeftIndices,
              Depth + 1);
    splitNode(Node->Right.get(), Data, rowLength, Labels, RightIndices,
              Depth + 1);
  }
}

// Split node function (using MAE)
void DecisionTreeSingle::splitNodeMAE(Tree *Node,
                                      const std::vector<double> &Data,
                                      int rowLength,
                                      const std::vector<double> &Labels,
                                      const std::vector<int> &Indices,
                                      int Depth) {
  Node->NodeMetric = Math::calculateMAEWithIndices(Labels, Indices);
  Node->NodeSamples = Indices.size();

  // Stopping conditions
  if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) ||
      Node->NodeMetric < MinError) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMedianWithIndices(Labels, Indices);
    return;
  }

  // Find the best split
  int BestFeature;
  double BestThreshold;
  double BestImpurityDecrease;

  if (numThreads < 1) {
    throw std::invalid_argument("numThreads must be >= 1");
  }

  // Find the best split

  if (useOMP) {
    std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
    findBestSplitUsingMAEOMP(Data, rowLength, Labels, Indices, Node->NodeMetric);
  } else {
    std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
    findBestSplitUsingMAE(Data, rowLength, Labels, Indices, Node->NodeMetric);
  }

  if (BestFeature == -1) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMedianWithIndices(Labels, Indices);
    return;
  }

  Node->FeatureIndex = BestFeature;
  Node->MaxValue = BestThreshold;

  // Split data
  std::vector<int> LeftIndices, RightIndices;
  for (int Idx : Indices) {
    if (Data[Idx * rowLength + BestFeature] <= BestThreshold) {
      LeftIndices.push_back(Idx);
    } else {
      RightIndices.push_back(Idx);
    }
  }

  Node->Left = std::make_unique<Tree>();
  Node->Right = std::make_unique<Tree>();

  if (LeftIndices.empty() || RightIndices.empty()) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMeanWithIndices(
        Labels, Indices); // Use mean instead of median
    return;
  }

  if (useOMP && Depth < maxSplitDepth) {
    // Grab the child pointers once ─ avoids capturing "Node" in each task
    auto *leftChild = Node->Left.get();
    auto *rightChild = Node->Right.get();

#pragma omp task shared(Data, Labels)                                          \
    firstprivate(leftChild, rowLength, LeftIndices, Depth)
    splitNodeMAE(leftChild, Data, rowLength, Labels, LeftIndices, Depth + 1);

#pragma omp task shared(Data, Labels)                                          \
    firstprivate(rightChild, rowLength, RightIndices, Depth)
    splitNodeMAE(rightChild, Data, rowLength, Labels, RightIndices, Depth + 1);

#pragma omp taskwait

  } else {
    // Perform normal sequential recursion after depth 2
    splitNodeMAE(Node->Left.get(), Data, rowLength, Labels, LeftIndices,
                 Depth + 1);
    splitNodeMAE(Node->Right.get(), Data, rowLength, Labels, RightIndices,
                 Depth + 1);
  }
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplit(
    const std::vector<double> &Data, int rowLength,
    const std::vector<double> &Labels, const std::vector<int> &Indices,
    double CurrentMSE) {

  int BestFeature = -1;
  double BestThreshold = 0.0;
  double BestImpurityDecrease = 0.0;

  size_t NumFeatures = rowLength;

  for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
    // Extract and sort feature values for the current feature
    std::vector<std::pair<double, int>> FeatureLabelPairs;
    for (int Idx : Indices) {
      FeatureLabelPairs.emplace_back(Data[Idx * rowLength + Feature], Idx);
    }

    std::sort(FeatureLabelPairs.begin(), FeatureLabelPairs.end());

    // Running sums for calculating MSE efficiently
    double LeftSum = 0.0, LeftSqSum = 0.0;
    size_t LeftCount = 0;
    double RightSum = 0.0, RightSqSum = 0.0;
    size_t RightCount = Indices.size();

    // Compute total right partition sum
    for (const auto &[value, Idx] : FeatureLabelPairs) {
      double Label = Labels[Idx];
      RightSum += Label;
      RightSqSum += Label * Label;
    }

    // Iterate over possible split points
    for (size_t i = 0; i < FeatureLabelPairs.size() - 1; ++i) {
      int Idx = FeatureLabelPairs[i].second;
      double Value = FeatureLabelPairs[i].first;
      double Label = Labels[Idx];

      // Update left partition
      LeftSum += Label;
      LeftSqSum += Label * Label;
      LeftCount++;

      // Update right partition
      RightSum -= Label;
      RightSqSum -= Label * Label;
      RightCount--;

      // Skip duplicate values
      double NextValue = FeatureLabelPairs[i + 1].first;
      if (Value == NextValue)
        continue;

      // Compute MSE for left and right partitions
      double LeftMean = LeftSum / LeftCount;
      double LeftMSE = (LeftSqSum - 2 * LeftMean * LeftSum +
                        LeftCount * LeftMean * LeftMean) /
                       LeftCount;

      double RightMean = RightSum / RightCount;
      double RightMSE = (RightSqSum - 2 * RightMean * RightSum +
                         RightCount * RightMean * RightMean) /
                        RightCount;

      // Calculate weighted impurity
      double WeightedImpurity =
          (LeftMSE * LeftCount + RightMSE * RightCount) / Indices.size();

      // Calculate impurity decrease
      double ImpurityDecrease = CurrentMSE - WeightedImpurity;

      // Update best split in a thread-safe way using OpenMP reduction
      if (ImpurityDecrease > BestImpurityDecrease) {
        BestImpurityDecrease = ImpurityDecrease;
        BestFeature = Feature;
        BestThreshold = (Value + NextValue) / 2.0;
      }
    }
  }

  return {BestFeature, BestThreshold, BestImpurityDecrease};
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitOMP(
    const std::vector<double> &Data, int rowLen,
    const std::vector<double> &Labels, const std::vector<int> &Indices,
    double CurrentMSE) {
  const size_t nFeat = rowLen;
  const size_t nSamp = Indices.size();

  BestSplit globalBest{0.0, -1, 0.0}; // reduction variable

#pragma omp parallel for reduction(maxBest : globalBest)                       \
    schedule(static) default(none)                                             \
    shared(Data, Labels, Indices, rowLen, nFeat, nSamp, CurrentMSE)
  for (size_t f = 0; f < nFeat; ++f) {
    std::vector<std::pair<double, int>> pairs;
    pairs.reserve(nSamp);
    for (int idx : Indices)
      pairs.emplace_back(Data[idx * rowLen + f], idx);

    std::sort(pairs.begin(), pairs.end());

    double lSum = 0.0, lSq = 0.0;
    size_t lCnt = 0;
    double rSum = 0.0, rSq = 0.0;
    size_t rCnt = nSamp;

    for (const auto &pr : pairs) {
      double y = Labels[pr.second];
      rSum += y;
      rSq += y * y;
    }

    for (size_t i = 0; i + 1 < pairs.size(); ++i) {
      const double x = pairs[i].first;
      const int idx = pairs[i].second;
      const double y = Labels[idx];
      const double xNext = pairs[i + 1].first;

      lSum += y;
      lSq += y * y;
      ++lCnt;
      rSum -= y;
      rSq -= y * y;
      --rCnt;

      if (x == xNext)
        continue; // skip duplicate split

      double lMean = lSum / lCnt;
      double rMean = rSum / rCnt;

      double lMSE = (lSq - 2 * lMean * lSum + lCnt * lMean * lMean) / lCnt;
      double rMSE = (rSq - 2 * rMean * rSum + rCnt * rMean * rMean) / rCnt;

      double weighted = (lMSE * lCnt + rMSE * rCnt) / nSamp;
      double impDec = CurrentMSE - weighted;

      if (impDec > globalBest.imp)
        globalBest = {impDec, static_cast<int>(f), 0.5 * (x + xNext)};
    }
  }

  return {globalBest.feat, globalBest.thr, globalBest.imp};
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitUsingMAE(
    const std::vector<double> &Data, int rowLength,
    const std::vector<double> &Labels, const std::vector<int> &Indices,
    double CurrentMAE) {

  // std::cout<<"findBestSplitUsingMAE called"<<std::endl;

  int BestFeature = -1;
  double BestThreshold = 0.0;
  double BestImpurityDecrease = 0.0;

  size_t NumFeatures = rowLength;

  for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
    // Store feature values and corresponding labels
    std::vector<std::pair<double, double>> FeatureLabelPairs;

    for (int Idx : Indices) {
      FeatureLabelPairs.emplace_back(Data[Idx * rowLength + Feature],
                                     Labels[Idx]);
    }

    // Sort by feature value
    std::sort(FeatureLabelPairs.begin(), FeatureLabelPairs.end());

    // Extract sorted labels
    std::vector<double> SortedLabels;
    for (const auto &[feat, label] : FeatureLabelPairs) {
      SortedLabels.push_back(label);
    }

    // Running sums and count
    double LeftSum = 0.0, RightSum = std::accumulate(SortedLabels.begin(),
                                                     SortedLabels.end(), 0.0);
    size_t LeftCount = 0, RightCount = SortedLabels.size();

    double LeftMedian = 0.0,
           RightMedian = Math::calculateMedian(SortedLabels); // Calculate once

    for (size_t i = 0; i < SortedLabels.size() - 1; ++i) {
      double Value = FeatureLabelPairs[i].first;
      double NextValue = FeatureLabelPairs[i + 1].first;
      double Label = FeatureLabelPairs[i].second;

      LeftSum += Label;
      RightSum -= Label;
      LeftCount++;
      RightCount--;

      if (Value == NextValue)
        continue; // Skip duplicate values

      // Update medians incrementally instead of recalculating
      LeftMedian = Math::incrementalMedian(SortedLabels, LeftCount);
      RightMedian = Math::incrementalMedian(SortedLabels, RightCount);

      // Compute MAE using sum instead of full iteration
      double LeftMAE = 0.0, RightMAE = 0.0;
      for (size_t j = 0; j < LeftCount; ++j) {
        LeftMAE += std::abs(SortedLabels[j] - LeftMedian);
      }
      for (size_t j = LeftCount; j < SortedLabels.size(); ++j) {
        RightMAE += std::abs(SortedLabels[j] - RightMedian);
      }

      double WeightedMAE = (LeftMAE + RightMAE) / SortedLabels.size();
      double ImpurityDecrease = CurrentMAE - WeightedMAE;

      if (ImpurityDecrease > BestImpurityDecrease) {
        BestImpurityDecrease = ImpurityDecrease;
        BestFeature = Feature;
        BestThreshold = (Value + NextValue) / 2.0;
      }
    }
  }

  return {BestFeature, BestThreshold, BestImpurityDecrease};
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitUsingMAEOMP(
    const std::vector<double> &Data, int rowLength,
    const std::vector<double> &Labels, const std::vector<int> &Indices,
    double CurrentMAE) {
  const size_t NumFeatures = rowLength;
  const size_t NumSamples = Indices.size();

  BestSplit globalBest{0.0, -1, 0.0}; // reduction variable

#pragma omp parallel for reduction(maxBest : globalBest) default(none) shared( \
        Data, Labels, Indices, rowLength, NumFeatures, CurrentMAE, NumSamples)
  for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {

    std::vector<std::pair<double, double>> col; // {x_i , y_i}
    col.reserve(NumSamples);
    for (int idx : Indices)
      col.emplace_back(Data[idx * rowLength + Feature], Labels[idx]);

    std::sort(col.begin(), col.end());

    std::vector<double> ySorted;
    ySorted.reserve(NumSamples);
    for (const auto &p : col)
      ySorted.push_back(p.second);

    size_t leftCnt = 0, rightCnt = NumSamples;
    double leftSum = 0.0,
           rightSum = std::accumulate(ySorted.begin(), ySorted.end(), 0.0);
    double leftMed = 0.0, rightMed = Math::calculateMedian(ySorted);

    BestSplit localBest{0.0, -1, 0.0};

    for (size_t i = 0; i + 1 < ySorted.size(); ++i) {
      const double x = col[i].first;
      const double xNext = col[i + 1].first;
      const double y = col[i].second;

      ++leftCnt;
      --rightCnt;
      leftSum += y;
      rightSum -= y;

      if (x == xNext)
        continue; // identical feature value

      leftMed = Math::incrementalMedian(ySorted, leftCnt);
      rightMed = Math::incrementalMedian(ySorted, rightCnt);

      /* compute MAE for each side ----------------------------- */
      double leftMAE = 0.0, rightMAE = 0.0;

      for (size_t j = 0; j < leftCnt; ++j)
        leftMAE += std::abs(ySorted[j] - leftMed);

      for (size_t j = leftCnt; j < ySorted.size(); ++j)
        rightMAE += std::abs(ySorted[j] - rightMed);

      const double weightedMAE = (leftMAE + rightMAE) / NumSamples;
      const double impurityDec = CurrentMAE - weightedMAE;

      if (impurityDec > globalBest.imp)
        globalBest = {impurityDec, static_cast<int>(Feature),
                      0.5 * (x + xNext)};
    }
  }
  return {globalBest.feat, globalBest.thr, globalBest.imp};
}

// Prediction function
double DecisionTreeSingle::predict(const double *Sample, int rowLength) const {
  const Tree *CurrentNode = Root.get();
  while (!CurrentNode->IsLeaf) {
    if (Sample[CurrentNode->FeatureIndex] <= CurrentNode->MaxValue) {
      CurrentNode = CurrentNode->Left.get();
    } else {
      CurrentNode = CurrentNode->Right.get();
    }
  }
  return CurrentNode->Prediction;
}

// Serialize node
void DecisionTreeSingle::serializeNode(const Tree *node, std::ostream &out) {
  if (!node) {
    out << "#\n"; // Mark empty node with "#"
    return;
  }

  // Write current node data, including NodeMetric and NodeSamples
  out << node->FeatureIndex << " " << node->MaxValue << " " << node->Prediction
      << " " << node->IsLeaf << " " << node->NodeMetric << " "
      << node->NodeSamples << "\n";

  // Recursively serialize left and right subtrees
  serializeNode(node->Left.get(), out);
  serializeNode(node->Right.get(), out);
}

// Deserialize node
std::unique_ptr<DecisionTreeSingle::Tree>
DecisionTreeSingle::deserializeNode(std::istream &in) {
  std::string line;
  std::getline(in, line);

  if (line == "#") {
    return nullptr;
  }

  auto node = std::make_unique<Tree>();
  std::istringstream iss(line);
  iss >> node->FeatureIndex >> node->MaxValue >> node->Prediction >>
      node->IsLeaf >> node->NodeMetric >> node->NodeSamples;

  node->Left = deserializeNode(in);
  node->Right = deserializeNode(in);

  return node;
}

// Tree save function
void DecisionTreeSingle::saveTree(const std::string &filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Write tree parameters
  out << MaxDepth << " " << MinLeafLarge << " " << MinError << " " << Criteria
      << " " << numThreads << " " << "\n";

  // Serialize the tree
  serializeNode(Root.get(), out);
  out.close();
}

// Load the tree
void DecisionTreeSingle::loadTree(const std::string &filename) {
  std::ifstream in(filename);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  // Read tree parameters
  in >> MaxDepth >> MinLeafLarge >> MinError >> Criteria >> numThreads;
  in.ignore(); // Ignore newline

  // Deserialize the tree
  Root = deserializeNode(in);
  in.close();
}

std::vector<char> DecisionTreeSingle::serializeToBuffer() {
  std::ostringstream oss(std::ios::binary);

  /*  write the two root statistics */
  oss.write(reinterpret_cast<const char *>(&RootSamples), sizeof(RootSamples));
  oss.write(reinterpret_cast<const char *>(&RootMSE), sizeof(RootMSE));

  /* write the whole tree topology */
  serializeNode(Root.get(), oss);

  const std::string &s = oss.str();
  return std::vector<char>(s.begin(), s.end());
}

std::unique_ptr<DecisionTreeSingle>
DecisionTreeSingle::deserializeFromBuffer(const std::vector<char> &buf) {
  std::istringstream iss(
      std::string(reinterpret_cast<const char *>(buf.data()), buf.size()),
      std::ios::binary);

  /* we need an instance; the default ctor must exist (can be empty) */
  auto tree = std::make_unique<DecisionTreeSingle>();

  /* ① restore the root statistics */
  iss.read(reinterpret_cast<char *>(&tree->RootSamples),
           sizeof(tree->RootSamples));
  iss.read(reinterpret_cast<char *>(&tree->RootMSE), sizeof(tree->RootMSE));

  /*  rebuild the topology */
  tree->Root = tree->deserializeNode(iss);

  return tree;
}

// Returns training parameters as a dictionnary
std::map<std::string, std::string>
DecisionTreeSingle::getTrainingParameters() const {
  std::map<std::string, std::string> parameters;
  parameters["MaxDepth"] = std::to_string(MaxDepth);
  parameters["MinLeafLarge"] = std::to_string(MinLeafLarge);
  parameters["MinError"] = std::to_string(MinError);
  parameters["Criteria"] = std::to_string(Criteria);
  parameters["UseOMP"] = useOMP ? "1.0" : "0.0";
  parameters["NumThreads"] = std::to_string(numThreads);
  return parameters;
}

// Retruns training parameters as a readable string
std::string DecisionTreeSingle::getTrainingParametersString() const {
  std::ostringstream oss;
  oss << "Training Parameters:\n";
  oss << "  - Max Depth: " << MaxDepth << "\n";
  oss << "  - Min Leaf Large: " << MinLeafLarge << "\n";
  oss << "  - Min Error: " << MinError << "\n";
  oss << "  - Criteria: " << (Criteria == 0 ? "MSE" : "MAE") << "\n";
  oss << "  - UseOMP: " << (useOMP ? "true" : "false") << "\n";
  oss << "  - Number of threads " << numThreads << "\n";
  return oss.str();
}