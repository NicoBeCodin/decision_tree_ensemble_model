#include "decision_tree_single.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// Constructor
DecisionTreeSingle::DecisionTreeSingle(int MaxDepth, int MinLeafLarge,
                                       double MinError, int Criteria,
                                       int numThreads)
    : MaxDepth(MaxDepth), MinLeafLarge(MinLeafLarge), MinError(MinError),
      Criteria(Criteria), Root(nullptr), numThreads(numThreads) {
        getMaxSplitDepth();
      }

// Training function
void DecisionTreeSingle::train(const std::vector<double> &Data, int rowLength,
                               const std::vector<double> &Labels,
                               int criteria) {
  Root = std::make_unique<Tree>();
  std::vector<int> Indices(Labels.size());
  std::iota(Indices.begin(), Indices.end(), 0);

  std::cout<<"Number of threads used is : "<<numThreads<<"\nTree will be split until "<<maxSplitDepth<<" depth"<<std::endl;
  // Will use MSE criterion
  if (criteria == 0) {
    std::cout << "Training using MSE." << std::endl;
    splitNode(Root.get(), Data, rowLength, Labels, Indices, 0);
  }
  // Will use MAE criterion
  else if (criteria == 1) {
    std::cout << "Training using MAE." << std::endl;
    splitNodeMAE(Root.get(), Data, rowLength, Labels, Indices, 0);
  }
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

  // Find the best split
  auto [BestFeature, BestThreshold, BestImpurityDecrease] =
      findBestSplit(Data, rowLength, Labels, Indices, Node->NodeMetric);

  if (BestFeature == -1) {
    Node->IsLeaf = true;
    Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices);
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

  // Parallelism goes here

  if (Depth < maxSplitDepth) { // Restrict parallelism to first two levels
    std::future<void> leftFuture =
        std::async(std::launch::async, &DecisionTreeSingle::splitNode, this,
                   Node->Left.get(), std::cref(Data), rowLength,
                   std::cref(Labels), std::cref(LeftIndices), Depth + 1);

    std::future<void> rightFuture =
        std::async(std::launch::async, &DecisionTreeSingle::splitNode, this,
                   Node->Right.get(), std::cref(Data), rowLength,
                   std::cref(Labels), std::cref(RightIndices), Depth + 1);

    // Wait for both subtrees to complete
    leftFuture.get();
    rightFuture.get();
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
  auto [BestFeature, BestThreshold, BestImpurityDecrease] =
      findBestSplitUsingMAE(Data, rowLength, Labels, Indices, Node->NodeMetric);

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
  

  if (Depth < maxSplitDepth) { // Restrict parallelism to first two levels
    std::future<void> leftFuture =
        std::async(std::launch::async, &DecisionTreeSingle::splitNodeMAE, this,
                   Node->Left.get(), std::cref(Data), rowLength,
                   std::cref(Labels), std::cref(LeftIndices), Depth + 1);

    std::future<void> rightFuture =
        std::async(std::launch::async, &DecisionTreeSingle::splitNodeMAE, this,
                   Node->Right.get(), std::cref(Data), rowLength,
                   std::cref(Labels), std::cref(RightIndices), Depth + 1);

    // Wait for both subtrees to complete
    leftFuture.get();
    rightFuture.get();
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
    std::vector<double> SortedValues;
    std::vector<int> SortedIndices;
    for (int Idx : Indices) {
      SortedValues.push_back(Data[Idx * rowLength + Feature]);
      SortedIndices.push_back(Idx);
    }
    std::sort(SortedIndices.begin(), SortedIndices.end(), [&](int A, int B) {
      return Data[A * rowLength + Feature] < Data[B * rowLength + Feature];
    });

    // Partition calculation for impurity
    double LeftSum = 0.0, LeftSqSum = 0.0;
    size_t LeftCount = 0;

    double RightSum = 0.0, RightSqSum = 0.0;
    size_t RightCount = Indices.size();
    for (int Idx : SortedIndices) {
      double Label = Labels[Idx];
      RightSum += Label;
      RightSqSum += Label * Label;
    }

    for (size_t i = 0; i < SortedIndices.size() - 1; ++i) {
      int Idx = SortedIndices[i];
      double Value = Data[Idx * rowLength + Feature];
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
      double NextValue = Data[SortedIndices[i + 1] * rowLength + Feature];
      if (Value == NextValue)
        continue;

      // Calculate MSE for left partition
      double LeftMean = LeftSum / LeftCount;
      double LeftMSE = (LeftSqSum - 2 * LeftMean * LeftSum +
                        LeftCount * LeftMean * LeftMean) /
                       LeftCount;

      // Calculate MSE for right partition
      double RightMean = RightSum / RightCount;
      double RightMSE = (RightSqSum - 2 * RightMean * RightSum +
                         RightCount * RightMean * RightMean) /
                        RightCount;

      // Calculate weighted impurity
      double WeightedImpurity =
          (LeftMSE * LeftCount + RightMSE * RightCount) / Indices.size();

      // Calculate impurity decrease
      double ImpurityDecrease = CurrentMSE - WeightedImpurity;

      // Update best split if current split is better
      if (ImpurityDecrease > BestImpurityDecrease) {
        BestImpurityDecrease = ImpurityDecrease;
        BestFeature = Feature;
        BestThreshold = (Value + NextValue) / 2.0;
      }
    }
  }

  return {BestFeature, BestThreshold, BestImpurityDecrease};
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitUsingMAE(
  const std::vector<double> &Data, int rowLength,
  const std::vector<double> &Labels, const std::vector<int> &Indices,
  double CurrentMAE) {

int BestFeature = -1;
double BestThreshold = 0.0;
double BestImpurityDecrease = 0.0;

size_t NumFeatures = rowLength;

for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
  // Store feature values and corresponding labels
  std::vector<std::pair<double, double>> FeatureLabelPairs;
  
  for (int Idx : Indices) {
    FeatureLabelPairs.emplace_back(Data[Idx * rowLength + Feature], Labels[Idx]);
  }

  // Sort by feature value
  std::sort(FeatureLabelPairs.begin(), FeatureLabelPairs.end());

  // Extract sorted labels
  std::vector<double> SortedLabels;
  for (const auto& [feat, label] : FeatureLabelPairs) {
    SortedLabels.push_back(label);
  }

  // Running sums and count
  double LeftSum = 0.0, RightSum = std::accumulate(SortedLabels.begin(), SortedLabels.end(), 0.0);
  size_t LeftCount = 0, RightCount = SortedLabels.size();

  double LeftMedian = 0.0, RightMedian = Math::calculateMedian(SortedLabels); // Calculate once

  for (size_t i = 0; i < SortedLabels.size() - 1; ++i) {
    double Value = FeatureLabelPairs[i].first;
    double NextValue = FeatureLabelPairs[i + 1].first;
    double Label = FeatureLabelPairs[i].second;

    LeftSum += Label;
    RightSum -= Label;
    LeftCount++;
    RightCount--;

    if (Value == NextValue) continue; // Skip duplicate values

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





// Other functions remain structurally similar with adjustments for flattened
// data

// Prediction function
double DecisionTreeSingle::predict(const std::vector<double> &Sample) const {
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

// Not necessary for the moment, just tryna merge
/*
    // Pre-sorted feature indices
    std::vector<std::vector<int>> DecisionTreeSingle::preSortFeatures(const
   std::vector<std::vector<double>>& Data, const std::vector<int>& Indices) {
        size_t NumFeatures = Data[0].size();
        std::vector<std::vector<int>> SortedIndices(NumFeatures, Indices);

        for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
            std::sort(SortedIndices[Feature].begin(),
   SortedIndices[Feature].end(),
                    [&Data, Feature](int A, int B) {
                        return Data[A][Feature] < Data[B][Feature];
                    });
        }
        return SortedIndices;
    }
*/

// Tree save function
void DecisionTreeSingle::saveTree(const std::string &filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Write tree parameters
  out << MaxDepth << " " << MinLeafLarge << " " << MinError << " " << Criteria
      << "\n";

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
  in >> MaxDepth >> MinLeafLarge >> MinError >> Criteria;
  in.ignore(); // Ignore newline

  // Deserialize the tree
  Root = deserializeNode(in);
  in.close();
}

// Returns training parameters as a dictionnary
std::map<std::string, std::string>
DecisionTreeSingle::getTrainingParameters() const {
  std::map<std::string, std::string> parameters;
  parameters["MaxDepth"] = std::to_string(MaxDepth);
  parameters["MinLeafLarge"] = std::to_string(MinLeafLarge);
  parameters["MinError"] = std::to_string(MinError);
  parameters["Criteria"] = std::to_string(Criteria);
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
  oss << "  - Number of threads " << numThreads << "\n";
  return oss.str();
}