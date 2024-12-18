#include "decision_tree_single.h"


#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// 构造函数
DecisionTreeSingle::DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError)
    : MaxDepth(MaxDepth), MinLeafLarge(MinLeafLarge), MinError(MinError), Root(nullptr) {}

// 训练函数
void DecisionTreeSingle::train(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, int criteria) {
    Root = std::make_unique<Tree>();
    std::vector<int> Indices(Data.size());
    std::iota(Indices.begin(), Indices.end(), 0);
    //Will use MSE
    if (criteria==0){
        splitNode(Root.get(), Data, Labels, Indices, 0);
    }
    //Will use MAE
    else if (criteria==1){
        splitNodeMAE(Root.get(), Data, Labels, Indices, 0);
    }
}

// 分裂节点函数
void DecisionTreeSingle::splitNode(Tree* Node, const std::vector<std::vector<double>>& Data,
                                   const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth) {
    // Calcul des métriques du nœud

    Node->NodeMetric = Math::calculateMSEWithIndices(Labels, Indices);
    
    Node->NodeSamples = Indices.size();

    // Conditions d'arrêt
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) || Node->NodeMetric < MinError) {
        Node->IsLeaf = true;
        Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices);
        return;
    }

    // Recherche du meilleur split
    auto [BestFeature, BestThreshold, BestImpurityDecrease] = findBestSplit(Data, Labels, Indices, Node->NodeMetric);

    if (BestFeature == -1) {
        Node->IsLeaf = true;
        Node->Prediction = Math::calculateMeanWithIndices(Labels, Indices);
        return;
    }

    Node->FeatureIndex = BestFeature;
    Node->MaxValue = BestThreshold;

    // Division des données
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


// 分裂节点函数

//SplitNode with MAE so currently testing
void DecisionTreeSingle::splitNodeMAE(Tree* Node, const std::vector<std::vector<double>>& Data,
                                   const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth) {
    

    
    Node->NodeMetric = Math::calculateMAEWithIndices(Labels, Indices);
    Node->NodeSamples = Indices.size();

    // Conditions d'arrêt
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) || Node->NodeMetric < MinError) {
        Node->IsLeaf = true;
        Node->Prediction = Math::calculateMedianWithIndices(Labels, Indices);
        return;
    }

    // Recherche du meilleur split
    auto [BestFeature, BestThreshold, BestImpurityDecrease] = findBestSplitUsingMAE(Data, Labels, Indices, Node->NodeMetric);

    if (BestFeature == -1) {
        Node->IsLeaf = true;
        Node->Prediction = Math::calculateMedianWithIndices(Labels, Indices);
        return;
    }

    Node->FeatureIndex = BestFeature;
    Node->MaxValue = BestThreshold;

    // Division des données
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


// 查找最佳分裂点
//Variance minimization
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



// 预测函数
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



// 预排序特征索引
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

// 序列化节点
void DecisionTreeSingle::serializeNode(const Tree* node, std::ostream& out) {
    if (!node) {
        out << "#\n"; // 用 "#" 标记空节点
        return;
    }

    // 写入当前节点的数据，包括 NodeMetric 和 NodeSamples
    out << node->FeatureIndex << " "
        << node->MaxValue << " "
        << node->Prediction << " "
        << node->IsLeaf << " "
        << node->NodeMetric << " "
        << node->NodeSamples << "\n";

    // 递归序列化左子树和右子树
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

        // Left and right partitions
        std::vector<double> LeftLabels, RightLabels;
        double LeftSum = 0.0, RightSum = 0.0;

        // Initialize right partition
        for (int idx : FeatureIndices) {
            RightLabels.push_back(Labels[idx]);
            RightSum += Labels[idx];
        }

        // Iterate over split candidates
        for (size_t i = 0; i < FeatureIndices.size() - 1; ++i) {
            int idx = FeatureIndices[i];
            double Value = Data[idx][Feature];
            double Label = Labels[idx];

            // Move label from right partition to left partition
            LeftLabels.push_back(Label);
            LeftSum += Label;

            RightSum -= Label;
            RightLabels.erase(std::remove(RightLabels.begin(), RightLabels.end(), Label), RightLabels.end());

            // Skip duplicates
            double NextValue = Data[FeatureIndices[i + 1]][Feature];
            if (Value == NextValue) continue;

            //Calculate medians which minimizes the MAE
            double LeftMedian = Math::calculateMedianSorted(LeftLabels);
            double RightMedian = Math::calculateMedianSorted(RightLabels);

            // Calculate MAE for left and right partitions
            double LeftMAE = Math::calculateMAE(LeftLabels, LeftMedian);
            double RightMAE = Math::calculateMAE(RightLabels, RightMedian);

            // Weighted MAE
            if (LeftLabels.empty() || RightLabels.empty()) continue;
            double WeightedMAE = (LeftMAE * LeftLabels.size() + RightMAE * RightLabels.size()) / Indices.size();
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



// 保存树
void DecisionTreeSingle::saveTree(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Écrire les paramètres de l'arbre
    out << MaxDepth << " " << MinLeafLarge << " " << MinError << "\n";
    
    // Sérialiser l'arbre
    serializeNode(Root.get(), out);
    out.close();
}

// 反序列化节点
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

// 加载树
void DecisionTreeSingle::loadTree(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Lire les paramètres de l'arbre
    in >> MaxDepth >> MinLeafLarge >> MinError;
    in.ignore(); // Ignorer le retour à la ligne
    
    // Désérialiser l'arbre
    Root = deserializeNode(in);
    in.close();
}
