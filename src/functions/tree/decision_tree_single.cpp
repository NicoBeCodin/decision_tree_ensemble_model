#include "decision_tree_single.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <tuple>

// Constructor
DecisionTreeSingle::DecisionTreeSingle(int MaxDepth, int MinLeafLarge,
                                       double MinError, int Criteria, bool useSplitHistogram,
                                       int numThreads)
    : MaxDepth(MaxDepth), MinLeafLarge(MinLeafLarge), MinError(MinError),
      Criteria(Criteria), useSplitHistogram(useSplitHistogram), Root(nullptr), numThreads(numThreads) {
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

  // Will use MSE criterion
  if (criteria == 0) {
    splitNode(Root.get(), Data, rowLength, Labels, Indices, 0);
  }
  // Will use MAE criterion
  else if (criteria == 1) {
    splitNodeMAE(Root.get(), Data, rowLength, Labels, Indices, 0);
  }
}

void DecisionTreeSingle::evaluate(const std::vector<double> &X_test, const int rowLength,
              const std::vector<double> &y_test, double &mse_value,
              double &mae_value)
{
  size_t test_size = y_test.size();
  std::vector<double> y_pred;
  y_pred.reserve(test_size);

  for (size_t i = 0; i < test_size; ++i) {
    const double* sample_ptr = &X_test[i * rowLength];
    y_pred.push_back(predict(sample_ptr, rowLength));
  }
  
  mse_value= Math::computeLossMSE(y_test, y_pred);
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
  double BestThreshold ;
  double BestImpurityDecrease;

  if (numThreads < 1) {
    throw std::invalid_argument("numThreads must be >= 1");
  }

  bool useOMP = numThreads > 1; // No need to set constructor argument

  // Find the best split (histogram or basic)
  if (useSplitHistogram) {
    if (useOMP) {
      std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
      findBestSplitHistogramOMP(Data, rowLength, Labels, Indices, Node->NodeMetric, 255);
    } 
    else {  
      std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
      findBestSplitHistogram(Data, rowLength, Labels, Indices, Node->NodeMetric, 255);
    }
  } else {
    if (useOMP) {
      std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
      findBestSplitOMP(Data, rowLength, Labels, Indices, Node->NodeMetric);
    } 
    else {  
      std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
      findBestSplit(Data, rowLength, Labels, Indices, Node->NodeMetric);
    }
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
  LeftIndices.reserve(Indices.size() / 2);  // reduces memory reallocations
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
    Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices); // Use mean instead of median
    return;
}

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
  int BestFeature;
  double BestThreshold;
  double BestImpurityDecrease;

  // Find the best split
  if (numThreads == 1) {
    std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
    findBestSplitHistogram(Data, rowLength, Labels, Indices, Node->NodeMetric, 255);
  } 
  else if (numThreads > 1) {  
    std::tie(BestFeature, BestThreshold, BestImpurityDecrease) =
    findBestSplitHistogramOMP(Data, rowLength, Labels, Indices, Node->NodeMetric, 255);
  }
  else {
    throw std::invalid_argument("numThreads must be >= 1");
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
    Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices); // Use mean instead of median
    return;
}

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
  
    // std::cout<<"findBestSplit called"<<std::endl;
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
          if (Value == NextValue) continue;

          // Compute MSE for left and right partitions
          double LeftMean = LeftSum / LeftCount;
          double LeftMSE = (LeftSqSum - 2 * LeftMean * LeftSum + LeftCount * LeftMean * LeftMean) / LeftCount;

          double RightMean = RightSum / RightCount;
          double RightMSE = (RightSqSum - 2 * RightMean * RightSum + RightCount * RightMean * RightMean) / RightCount;

          // Calculate weighted impurity
          double WeightedImpurity = (LeftMSE * LeftCount + RightMSE * RightCount) / Indices.size();

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
  const std::vector<double> &Data, int rowLength,
  const std::vector<double> &Labels, const std::vector<int> &Indices,
  double CurrentMSE) {

// Shared best variables
int BestFeature = -1;
double BestThreshold = 0.0;
double BestImpurityDecrease = 0.0;

size_t NumFeatures = rowLength;
size_t NumSamples = Indices.size();  

if (numThreads != 1) {
  omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
}

// Thread-private best values
#pragma omp parallel 
{
  int ThreadBestFeature = -1;
  double ThreadBestThreshold = 0.0;
  double ThreadBestImpurityDecrease = 0.0;

  #pragma omp for nowait
  for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
    std::vector<std::pair<double, int>> FeatureLabelPairs;
    
    for (int Idx : Indices) {
      FeatureLabelPairs.emplace_back(Data[Idx * rowLength + Feature], Idx);
    }
    std::sort(FeatureLabelPairs.begin(), FeatureLabelPairs.end());

    double LeftSum = 0.0, LeftSqSum = 0.0;
    size_t LeftCount = 0;
    double RightSum = 0.0, RightSqSum = 0.0;
    size_t RightCount = NumSamples;

    for (const auto &[value, Idx] : FeatureLabelPairs) {
      double Label = Labels[Idx];
      RightSum += Label;
      RightSqSum += Label * Label;
    }

    for (size_t i = 0; i < FeatureLabelPairs.size() - 1; ++i) {
      int Idx = FeatureLabelPairs[i].second;
      double Value = FeatureLabelPairs[i].first;
      double Label = Labels[Idx];

      LeftSum += Label;
      LeftSqSum += Label * Label;
      LeftCount++;

      RightSum -= Label;
      RightSqSum -= Label * Label;
      RightCount--;

      double NextValue = FeatureLabelPairs[i + 1].first;
      if (Value == NextValue) continue;

      double LeftMean = LeftSum / LeftCount;
      double LeftMSE = (LeftSqSum - 2 * LeftMean * LeftSum + LeftCount * LeftMean * LeftMean) / LeftCount;

      double RightMean = RightSum / RightCount;
      double RightMSE = (RightSqSum - 2 * RightMean * RightSum + RightCount * RightMean * RightMean) / RightCount;

      double WeightedImpurity = (LeftMSE * LeftCount + RightMSE * RightCount) / NumSamples;
      double ImpurityDecrease = CurrentMSE - WeightedImpurity;

      if (ImpurityDecrease > ThreadBestImpurityDecrease) {
        ThreadBestImpurityDecrease = ImpurityDecrease;
        ThreadBestFeature = Feature;
        ThreadBestThreshold = (Value + NextValue) / 2.0;
      }
    }
  }

  // Safely update global best using a critical section
  #pragma omp critical
  {
    if (ThreadBestImpurityDecrease > BestImpurityDecrease) {
      BestImpurityDecrease = ThreadBestImpurityDecrease;
      BestFeature = ThreadBestFeature;
      BestThreshold = ThreadBestThreshold;
    }
  }
}

return {BestFeature, BestThreshold, BestImpurityDecrease};
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

int BestFeature = -1;  // ✅ Corrected initialization
double BestThreshold = 0.0;
double BestImpurityDecrease = 0.0;

// std::cout<<"findBestSplitUsingMAEOMP called"<<std::endl;

size_t NumFeatures = rowLength;

if (numThreads !=1) {
  omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));
}

#pragma omp parallel for default(none) shared(Data, Labels, Indices, NumFeatures, rowLength, CurrentMAE, BestFeature, BestThreshold) \
    reduction(max : BestImpurityDecrease)
for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
  
  // ✅ Store feature values and corresponding labels (private per thread)
  std::vector<std::pair<double, double>> FeatureLabelPairs;
  for (int Idx : Indices) {
    FeatureLabelPairs.emplace_back(Data[Idx * rowLength + Feature], Labels[Idx]);
  }

  std::sort(FeatureLabelPairs.begin(), FeatureLabelPairs.end());  // ✅ Sorting stays private

  // Extract sorted labels
  std::vector<double> SortedLabels;
  for (const auto &[feat, label] : FeatureLabelPairs) {
    SortedLabels.push_back(label);
  }

  double LeftSum = 0.0, RightSum = std::accumulate(SortedLabels.begin(), SortedLabels.end(), 0.0);
  size_t LeftCount = 0, RightCount = SortedLabels.size();

  double LeftMedian = 0.0, RightMedian = Math::calculateMedian(SortedLabels);

  for (size_t i = 0; i < SortedLabels.size() - 1; ++i) {
    double Value = FeatureLabelPairs[i].first;
    double NextValue = FeatureLabelPairs[i + 1].first;
    double Label = FeatureLabelPairs[i].second;

    LeftSum += Label;
    RightSum -= Label;
    LeftCount++;
    RightCount--;

    if (Value == NextValue) continue;

    LeftMedian = Math::incrementalMedian(SortedLabels, LeftCount);
    RightMedian = Math::incrementalMedian(SortedLabels, RightCount);

    double LeftMAE = 0.0, RightMAE = 0.0;
    for (size_t j = 0; j < LeftCount; ++j) {
      LeftMAE += std::abs(SortedLabels[j] - LeftMedian);
    }
    for (size_t j = LeftCount; j < SortedLabels.size(); ++j) {
      RightMAE += std::abs(SortedLabels[j] - RightMedian);
    }

    double WeightedMAE = (LeftMAE + RightMAE) / SortedLabels.size();
    // #pragma omp ordered
    double ImpurityDecrease = CurrentMAE - WeightedMAE;

    // ✅ Thread-safe update of best split
    #pragma omp critical
    {
      if (ImpurityDecrease > BestImpurityDecrease) {
        BestImpurityDecrease = ImpurityDecrease;
        BestFeature = Feature;  // ✅ Fix incorrect OpenMP reduction
        BestThreshold = (Value + NextValue) / 2.0;
      }
    }
  }
}

// std::cout<<"New vals for findBestSplitUsingMAEOMP best impurity decrease: " << BestImpurityDecrease 
//          << " bestFeature " << BestFeature << " BestThreshold " << BestThreshold << std::endl;

return {BestFeature, BestThreshold, BestImpurityDecrease};
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitHistogram(
  const std::vector<double>& Data,
  int rowLength,
  const std::vector<double>& Labels,
  const std::vector<int>& Indices,
  double CurrentMSE,
  int n_bins = 255) {

  int BestFeature = -1;
  double BestThreshold = 0.0;
  double BestImpurityDecrease = 0.0;

  size_t n_features = rowLength;
  size_t n_samples = Indices.size();

  // Step 1 : Precompute min and max per feature
  std::vector<std::pair<double, double>> feature_min_max(n_features, {1e9, -1e9});

  for (size_t idx : Indices) {
      for (size_t f = 0; f < n_features; ++f) {
          double val = Data[idx * rowLength + f];
          feature_min_max[f].first = std::min(feature_min_max[f].first, val);
          feature_min_max[f].second = std::max(feature_min_max[f].second, val);
      }
  }

  // Step 2 : For each feature, build histogram
  for (size_t f = 0; f < n_features; ++f) {
      double fmin = feature_min_max[f].first;
      double fmax = feature_min_max[f].second;

      if (fmin == fmax) continue; // Skip constant features

      std::vector<double> sum_in_bin(n_bins, 0.0);
      std::vector<double> sum_sq_in_bin(n_bins, 0.0);
      std::vector<int> count_in_bin(n_bins, 0);

      // Fill histogram
      for (size_t idx : Indices) {
          double val = Data[idx * rowLength + f];
          int bin = static_cast<int>(((val - fmin) / (fmax - fmin)) * (n_bins - 1));
          bin = std::max(0, std::min(bin, n_bins - 1)); // Safety
          double label = Labels[idx];
          sum_in_bin[bin] += label;
          sum_sq_in_bin[bin] += label * label;
          count_in_bin[bin] += 1;
      }

      // Step 3: Find best split by scanning bins
      double left_sum = 0.0, left_sq_sum = 0.0;
      int left_count = 0;

      double right_sum = std::accumulate(sum_in_bin.begin(), sum_in_bin.end(), 0.0);
      double right_sq_sum = std::accumulate(sum_sq_in_bin.begin(), sum_sq_in_bin.end(), 0.0);
      int right_count = std::accumulate(count_in_bin.begin(), count_in_bin.end(), 0);

      for (int b = 0; b < n_bins - 1; ++b) {  // Split between bin b and b+1
          left_sum += sum_in_bin[b];
          left_sq_sum += sum_sq_in_bin[b];
          left_count += count_in_bin[b];

          right_sum -= sum_in_bin[b];
          right_sq_sum -= sum_sq_in_bin[b];
          right_count -= count_in_bin[b];

          if (left_count == 0 || right_count == 0) continue;

          double left_mean = left_sum / left_count;
          double left_mse = (left_sq_sum - 2 * left_mean * left_sum + left_count * left_mean * left_mean) / left_count;

          double right_mean = right_sum / right_count;
          double right_mse = (right_sq_sum - 2 * right_mean * right_sum + right_count * right_mean * right_mean) / right_count;

          double weighted_mse = (left_mse * left_count + right_mse * right_count) / n_samples;
          double impurity_decrease = CurrentMSE - weighted_mse;

          if (impurity_decrease > BestImpurityDecrease) {
              BestImpurityDecrease = impurity_decrease;
              BestFeature = f;
              // Find threshold value corresponding to bin boundary
              BestThreshold = fmin + ((fmax - fmin) * (b + 1)) / n_bins;
          }
      }
  }

  return {BestFeature, BestThreshold, BestImpurityDecrease};
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitHistogramOMP(
  const std::vector<double>& Data,
  int rowLength,
  const std::vector<double>& Labels,
  const std::vector<int>& Indices,
  double CurrentMSE,
  int n_bins = 255) {

  int BestFeature = -1;
  double BestThreshold = 0.0;
  double BestImpurityDecrease = 0.0;

  size_t n_features = rowLength;
  size_t n_samples = Indices.size();

  if (numThreads != 1) {
    omp_set_num_threads(std::max(1, omp_get_max_threads()));
  }

  // Precompute min and max for each feature
  std::vector<std::pair<double, double>> feature_min_max(n_features, {1e9, -1e9});
  for (size_t idx : Indices) {
      for (size_t f = 0; f < n_features; ++f) {
          double val = Data[idx * rowLength + f];
          feature_min_max[f].first = std::min(feature_min_max[f].first, val);
          feature_min_max[f].second = std::max(feature_min_max[f].second, val);
      }
  }

  // Parallel over features
  #pragma omp parallel
  {
      int thread_best_feature = -1;
      double thread_best_threshold = 0.0;
      double thread_best_impurity_decrease = 0.0;

      #pragma omp for nowait
      for (int f = 0; f < static_cast<int>(n_features); ++f) {
          double fmin = feature_min_max[f].first;
          double fmax = feature_min_max[f].second;

          if (fmin == fmax) continue; // Skip constant features

          std::vector<double> sum_in_bin(n_bins, 0.0);
          std::vector<double> sum_sq_in_bin(n_bins, 0.0);
          std::vector<int> count_in_bin(n_bins, 0);

          // Fill histogram for feature f
          for (size_t idx : Indices) {
              double val = Data[idx * rowLength + f];
              int bin = static_cast<int>(((val - fmin) / (fmax - fmin)) * (n_bins - 1));
              bin = std::max(0, std::min(bin, n_bins - 1)); // Clamp
              double label = Labels[idx];
              sum_in_bin[bin] += label;
              sum_sq_in_bin[bin] += label * label;
              count_in_bin[bin] += 1;
          }

          double left_sum = 0.0, left_sq_sum = 0.0;
          int left_count = 0;
          double right_sum = std::accumulate(sum_in_bin.begin(), sum_in_bin.end(), 0.0);
          double right_sq_sum = std::accumulate(sum_sq_in_bin.begin(), sum_sq_in_bin.end(), 0.0);
          int right_count = std::accumulate(count_in_bin.begin(), count_in_bin.end(), 0);

          for (int b = 0; b < n_bins - 1; ++b) {
              left_sum += sum_in_bin[b];
              left_sq_sum += sum_sq_in_bin[b];
              left_count += count_in_bin[b];

              right_sum -= sum_in_bin[b];
              right_sq_sum -= sum_sq_in_bin[b];
              right_count -= count_in_bin[b];

              if (left_count == 0 || right_count == 0) continue;

              double left_mean = left_sum / left_count;
              double left_mse = (left_sq_sum - 2 * left_mean * left_sum + left_count * left_mean * left_mean) / left_count;

              double right_mean = right_sum / right_count;
              double right_mse = (right_sq_sum - 2 * right_mean * right_sum + right_count * right_mean * right_mean) / right_count;

              double weighted_mse = (left_mse * left_count + right_mse * right_count) / n_samples;
              double impurity_decrease = CurrentMSE - weighted_mse;

              if (impurity_decrease > thread_best_impurity_decrease) {
                  thread_best_impurity_decrease = impurity_decrease;
                  thread_best_feature = f;
                  thread_best_threshold = fmin + ((fmax - fmin) * (b + 1)) / n_bins;
              }
          }
      }

      // Update global best
      #pragma omp critical
      {
          if (thread_best_impurity_decrease > BestImpurityDecrease) {
              BestImpurityDecrease = thread_best_impurity_decrease;
              BestFeature = thread_best_feature;
              BestThreshold = thread_best_threshold;
          }
      }
  }

  return {BestFeature, BestThreshold, BestImpurityDecrease};
}

// Other functions remain structurally similar with adjustments for flattened
// data

// Prediction function
double DecisionTreeSingle::predict(const double* Sample, int rowLength) const {
  const Tree* CurrentNode = Root.get();
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
      << useSplitHistogram << " " << numThreads << " " << "\n";

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
  in >> MaxDepth >> MinLeafLarge >> MinError >> Criteria >> useSplitHistogram >> numThreads;
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
  parameters["UseSplitHistogram"] = useSplitHistogram ? "1.0" : "0.0";
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
  oss << "  - UseSplitHistogram: " << (useSplitHistogram ? "true" : "false") << "\n";
  oss << "  - Number of threads " << numThreads << "\n";
  return oss.str();
}