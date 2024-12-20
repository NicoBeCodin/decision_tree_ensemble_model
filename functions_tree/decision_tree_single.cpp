#include "decision_tree_single.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// Constructor
DecisionTreeSingle::DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError)
    : MaxDepth(MaxDepth), MinLeafLarge(MinLeafLarge), MinError(MinError), Root(nullptr) {}

// Training function
void DecisionTreeSingle::train(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, int criteria) {
    Root = std::make_unique<Tree>();
    std::vector<int> Indices(Data.size());
    std::iota(Indices.begin(), Indices.end(), 0);
    // Will use MSE
    if (criteria == 0) {
        splitNode(Root.get(), Data, Labels, Indices, 0);
    }
    // Will use MAE
    else if (criteria == 1) {
        splitNodeMAE(Root.get(), Data, Labels, Indices, 0);
    }
}

// Split node function
void DecisionTreeSingle::splitNode(Tree* Node, const std::vector<std::vector<double>>& Data,
                                   const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth) {
    // Compute node metrics

    Node->NodeMetric = Math::calculateMSEWithIndices(Labels, Indices);
    
    Node->NodeSamples = Indices.size();

    // Stopping conditions
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) || Node->NodeMetric < MinError) {
        Node->IsLeaf = true;
        Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices);
        return;
    }

    // Find the best split
    auto [BestFeature, BestThreshold, BestImpurityDecrease] = findBestSplit(Data, Labels, Indices, Node->NodeMetric);

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
        if (Data[Idx][BestFeature] <= BestThreshold) {
            LeftIndices.push_back(Idx);
        } else {
            RightIndices.push_back(Idx);
        }
    }

    Node->Left = std::make_unique<Tree>();
    Node->Right = std::make_unique<Tree>();
    splitNode(Node->Left.get(), Data, Labels, LeftIndices, Depth + 1);
    splitNode(Node->Right.get(), Data, Labels, RightIndices, Depth + 1);
}

// SplitNode with MAE
void DecisionTreeSingle::splitNodeMAE(Tree* Node, const std::vector<std::vector<double>>& Data,
                                      const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth) {
    Node->NodeMetric = Math::calculateMAEWithIndices(Labels, Indices);
    Node->NodeSamples = Indices.size();

    // Stopping conditions
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) || Node->NodeMetric < MinError) {
        Node->IsLeaf = true;
        Node->Prediction = Math::calculateMedianWithIndices(Labels, Indices);
        return;
    }

    // Find the best split
    auto [BestFeature, BestThreshold, BestImpurityDecrease] = findBestSplitUsingMAE(Data, Labels, Indices, Node->NodeMetric);

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
        if (Data[Idx][BestFeature] <= BestThreshold) {
            LeftIndices.push_back(Idx);
        } else {
            RightIndices.push_back(Idx);
        }
    }

    Node->Left = std::make_unique<Tree>();
    Node->Right = std::make_unique<Tree>();
    splitNode(Node->Left.get(), Data, Labels, LeftIndices, Depth + 1);
    splitNode(Node->Right.get(), Data, Labels, RightIndices, Depth + 1);
}

// Find the best split (variance minimization)
std::tuple<int, double, double> DecisionTreeSingle::findBestSplit(const std::vector<std::vector<double>>& Data,
                                                                  const std::vector<double>& Labels, const std::vector<int>& Indices, double CurrentMSE) {
    int BestFeature = -1;
    double BestThreshold = 0.0;
    double BestImpurityDecrease = 0.0;

    size_t NumFeatures = Data[0].size();
    auto SortedFeatureIndices = preSortFeatures(Data, Indices);

    for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
        const auto& FeatureIndices = SortedFeatureIndices[Feature];

        double LeftSum = 0.0, LeftSqSum = 0.0;
        size_t LeftCount = 0;

        double RightSum = 0.0, RightSqSum = 0.0;
        size_t RightCount = Indices.size();
        for (int Idx : FeatureIndices) {
            double Label = Labels[Idx];
            RightSum += Label;
            RightSqSum += Label * Label;
        }

        for (size_t i = 0; i < FeatureIndices.size() - 1; ++i) {
            int Idx = FeatureIndices[i];
            double Value = Data[Idx][Feature];
            double Label = Labels[Idx];

            LeftSum += Label;
            LeftSqSum += Label * Label;
            LeftCount++;

            RightSum -= Label;
            RightSqSum -= Label * Label;
            RightCount--;

            double NextValue = Data[FeatureIndices[i + 1]][Feature];
            if (Value == NextValue) continue;

            double LeftMean = LeftSum / LeftCount;
            double LeftMSE = (LeftSqSum - 2 * LeftMean * LeftSum + LeftCount * LeftMean * LeftMean) / LeftCount;

            double RightMean = RightSum / RightCount;
            double RightMSE = (RightSqSum - 2 * RightMean * RightSum + RightCount * RightMean * RightMean) / RightCount;

            double WeightedImpurity = (LeftMSE * LeftCount + RightMSE * RightCount) / Indices.size();
            double ImpurityDecrease = CurrentMSE - WeightedImpurity;

            if (ImpurityDecrease > BestImpurityDecrease) {
                BestImpurityDecrease = ImpurityDecrease;
                BestFeature = Feature;
                BestThreshold = (Value + NextValue) / 2.0;
            }
        }
    }
    return {BestFeature, BestThreshold, BestImpurityDecrease};
}

// Predict function
double DecisionTreeSingle::predict(const std::vector<double>& Sample) const {
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

// Pre-sorted feature indices
std::vector<std::vector<int>> DecisionTreeSingle::preSortFeatures(const std::vector<std::vector<double>>& Data, const std::vector<int>& Indices) {
    size_t NumFeatures = Data[0].size();
    std::vector<std::vector<int>> SortedIndices(NumFeatures, Indices);

    for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
        std::sort(SortedIndices[Feature].begin(), SortedIndices[Feature].end(),
                  [&Data, Feature](int A, int B) {
                      return Data[A][Feature] < Data[B][Feature];
                  });
    }
    return SortedIndices;
}

// Serialize node
void DecisionTreeSingle::serializeNode(const Tree* node, std::ostream& out) {
    if (!node) {
        out << "#\n"; // Mark empty node with "#"
        return;
    }

    // Write current node data, including NodeMetric and NodeSamples
    out << node->FeatureIndex << " "
        << node->MaxValue << " "
        << node->Prediction << " "
        << node->IsLeaf << " "
        << node->NodeMetric << " "
        << node->NodeSamples << "\n";

    // Recursively serialize left and right subtrees
    serializeNode(node->Left.get(), out);
    serializeNode(node->Right.get(), out);
}

std::tuple<int, double, double> DecisionTreeSingle::findBestSplitUsingMAE(
    const std::vector<std::vector<double>>& Data,
    const std::vector<double>& Labels,
    const std::vector<int>& Indices,
    double CurrentMAE) 
{
    int BestFeature = -1;
    double BestThreshold = 0.0;
    double BestImpurityDecrease = 0.0;

    size_t NumFeatures = Data[0].size();
    auto SortedFeatureIndices = preSortFeatures(Data, Indices);

    for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
        const auto& FeatureIndices = SortedFeatureIndices[Feature];

        // Prepare cumulative counts and sums
        std::vector<double> SortedLabels;
        for (int idx : FeatureIndices) {
            SortedLabels.push_back(Labels[idx]);
        }
        std::sort(SortedLabels.begin(), SortedLabels.end());

        double LeftSum = 0.0, RightSum = std::accumulate(SortedLabels.begin(), SortedLabels.end(), 0.0);
        size_t LeftCount = 0, RightCount = SortedLabels.size();

        // Iterate over split candidates
        for (size_t i = 0; i < FeatureIndices.size() - 1; ++i) {
            int idx = FeatureIndices[i];
            double Value = Data[idx][Feature];

            // Update left and right partitions
            LeftSum += Labels[idx];
            RightSum -= Labels[idx];
            LeftCount++;
            RightCount--;

            // Skip duplicates
            double NextValue = Data[FeatureIndices[i + 1]][Feature];
            if (Value == NextValue) continue;

            // Calculate medians directly from sorted labels
            double LeftMedian = SortedLabels[LeftCount / 2];
            if (LeftCount % 2 == 0) {
                LeftMedian = (SortedLabels[LeftCount / 2 - 1] + SortedLabels[LeftCount / 2]) / 2.0;
            }

            double RightMedian = SortedLabels[LeftCount + RightCount / 2];
            if (RightCount % 2 == 0) {
                RightMedian = (SortedLabels[LeftCount + RightCount / 2 - 1] + SortedLabels[LeftCount + RightCount / 2]) / 2.0;
            }

            // Calculate MAE for left and right partitions
            double LeftMAE = 0.0, RightMAE = 0.0;
            for (size_t j = 0; j < LeftCount; ++j) {
                LeftMAE += std::abs(SortedLabels[j] - LeftMedian);
            }
            for (size_t j = LeftCount; j < SortedLabels.size(); ++j) {
                RightMAE += std::abs(SortedLabels[j] - RightMedian);
            }

            double WeightedMAE = (LeftMAE + RightMAE) / SortedLabels.size();
            double ImpurityDecrease = CurrentMAE - WeightedMAE;

            // Update best split
            if (ImpurityDecrease > BestImpurityDecrease) {
                BestImpurityDecrease = ImpurityDecrease;
                BestFeature = Feature;
                BestThreshold = (Value + NextValue) / 2.0;
            }
        }
    }

    return {BestFeature, BestThreshold, BestImpurityDecrease};
}

// Save the tree
void DecisionTreeSingle::saveTree(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write tree parameters
    out << MaxDepth << " " << MinLeafLarge << " " << MinError << " " << Criteria << "\n";
    
    // Serialize the tree
    serializeNode(Root.get(), out);
    out.close();
}

// Deserialize node
std::unique_ptr<DecisionTreeSingle::Tree> DecisionTreeSingle::deserializeNode(std::istream& in) {
    std::string line;
    std::getline(in, line);
    
    if (line == "#") {
        return nullptr;
    }

    auto node = std::make_unique<Tree>();
    std::istringstream iss(line);
    iss >> node->FeatureIndex
        >> node->MaxValue
        >> node->Prediction
        >> node->IsLeaf
        >> node->NodeMetric
        >> node->NodeSamples;

    node->Left = deserializeNode(in);
    node->Right = deserializeNode(in);

    return node;
}

// Load the tree
void DecisionTreeSingle::loadTree(const std::string& filename) {
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

// Retourne les paramètres d'entraînement sous forme de dictionnaire (clé-valeur)
std::map<std::string, std::string> DecisionTreeSingle::getTrainingParameters() const {
    std::map<std::string, std::string> parameters;
    parameters["MaxDepth"] = std::to_string(MaxDepth);
    parameters["MinLeafLarge"] = std::to_string(MinLeafLarge);
    parameters["MinError"] = std::to_string(MinError);
    parameters["Criteria"] = std::to_string(Criteria);
    return parameters;
}

// Retourne les paramètres d'entraînement sous forme d'une chaîne de caractères lisible
std::string DecisionTreeSingle::getTrainingParametersString() const {
    std::ostringstream oss;
    oss << "Training Parameters:\n";
    oss << "  - Max Depth: " << MaxDepth << "\n";
    oss << "  - Min Leaf Large: " << MinLeafLarge << "\n";
    oss << "  - Min Error: " << MinError << "\n";
    oss << "  - Criteria: " << (Criteria == 0 ? "MSE" : "MAE") << "\n";
    return oss.str();
}
