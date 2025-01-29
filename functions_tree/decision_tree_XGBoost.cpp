#include "decision_tree_XGBoost.h"

/**
 * @brief Constructor to initialize the decision tree with specified parameters
 * @param MaxDepth Maximum depth of the tree
 * @param MinLeafSize Minimum size of a leaf
 * @param Lambda L2 regularization parameter
 * @param Gamma Minimum gain threshold for a split
 */
DecisionTreeXGBoost::DecisionTreeXGBoost(int MaxDepth, int MinLeafSize, double Lambda, double Gamma)
    : MaxDepth(MaxDepth), MinLeafSize(MinLeafSize), Lambda(Lambda), Gamma(Gamma), Root(nullptr) {}

/**
 * @brief Train the decision tree
 * @param Data Flattened training feature matrix (1D vector)
 * @param rowLength Number of features per row/sample
 * @param Labels Target labels vector
 * @param Predictions Current predictions vector (used to calculate gradients)
 */
void DecisionTreeXGBoost::train(const std::vector<double>& Data, int rowLength, 
                                const std::vector<double>& Labels, 
                                std::vector<double>& Predictions) {
    Root = std::make_unique<Tree>();
    std::vector<int> Indices(Labels.size());
    std::iota(Indices.begin(), Indices.end(), 0);

    // Compute gradients and hessians
    std::vector<double> Gradients(Labels.size()), Hessians(Labels.size());
    computeGradientsAndHessians(Labels, Predictions, Gradients, Hessians);

    // Build the tree
    splitNode(Root.get(), Data, rowLength, Gradients, Hessians, Indices, 0);
}

/**
 * @brief Predict the value for a single sample
 * @param Sample Feature vector representing a single sample
 * @return Prediction made by the tree for this sample
 */
double DecisionTreeXGBoost::predict(const std::vector<double>& Sample) const {
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

/**
 * @brief Compute gradients and hessians for each sample
 * @param Labels Vector of true target labels
 * @param Predictions Vector of current predictions
 * @param Gradients Output vector containing calculated gradients
 * @param Hessians Output vector containing calculated hessians
 */
void DecisionTreeXGBoost::computeGradientsAndHessians(
    const std::vector<double>& Labels,
    const std::vector<double>& Predictions,
    std::vector<double>& Gradients,
    std::vector<double>& Hessians) {
    for (size_t i = 0; i < Labels.size(); ++i) {
        double pred = Predictions[i];
        double label = Labels[i];
        Gradients[i] = 2 * (pred - label); // Gradient of MSE
        Hessians[i] = 2.0;                // Constant hessian for MSE
    }
}

/**
 * @brief Split a node in the tree
 * @param Node Pointer to the node to be split
 * @param Data Flattened feature matrix (1D vector)
 * @param rowLength Number of features per row/sample
 * @param Gradients Gradients vector
 * @param Hessians Hessians vector
 * @param Indices Indices of samples available for this split
 * @param Depth Current depth of the node
 */
void DecisionTreeXGBoost::splitNode(Tree* Node, const std::vector<double>& Data, int rowLength, 
                                    const std::vector<double>& Gradients,
                                    const std::vector<double>& Hessians,
                                    const std::vector<int>& Indices, int Depth) {
    // Stopping criteria
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafSize)) {
        Node->IsLeaf = true;
        Node->Prediction = calculateLeafWeight(Indices, Gradients, Hessians);
        return;
    }

    // Find the best split
    auto [BestFeature, BestThreshold, BestGain] = findBestSplit(Data, rowLength, Gradients, Hessians, Indices);

    if (BestFeature == -1 || BestGain < Gamma) { // No significant gain
        Node->IsLeaf = true;
        Node->Prediction = calculateLeafWeight(Indices, Gradients, Hessians);
        return;
    }

    Node->FeatureIndex = BestFeature;
    Node->MaxValue = BestThreshold;
    Node->GainImprovement = BestGain;

    // Partition indices
    std::vector<int> LeftIndices, RightIndices;
    for (int idx : Indices) {
        if (Data[idx * rowLength + BestFeature] <= BestThreshold) {
            LeftIndices.push_back(idx);
        } else {
            RightIndices.push_back(idx);
        }
    }

    Node->Left = std::make_unique<Tree>();
    Node->Right = std::make_unique<Tree>();
    splitNode(Node->Left.get(), Data, rowLength, Gradients, Hessians, LeftIndices, Depth + 1);
    splitNode(Node->Right.get(), Data, rowLength, Gradients, Hessians, RightIndices, Depth + 1);
}

/**
 * @brief Find the best possible split for a node
 * @param Data Flattened feature matrix (1D vector)
 * @param rowLength Number of features per row/sample
 * @param Gradients Gradients vector
 * @param Hessians Hessians vector
 * @param Indices Indices of samples available for this split
 * @return Tuple containing the best feature, threshold, and gain
 */
std::tuple<int, double, double> DecisionTreeXGBoost::findBestSplit(
    const std::vector<double>& Data, int rowLength,
    const std::vector<double>& Gradients,
    const std::vector<double>& Hessians,
    const std::vector<int>& Indices) {
    
    int BestFeature = -1;
    double BestThreshold = 0.0;
    double BestGain = 0.0;

    double G = sumGradients(Gradients, Indices);
    double H = sumHessians(Hessians, Indices);
    double CurrentScore = G * G / (H + Lambda);

    for (int feature = 0; feature < rowLength; ++feature) {
        std::vector<std::pair<double, int>> SortedValues;
        for (int idx : Indices) {
            SortedValues.emplace_back(Data[idx * rowLength + feature], idx);
        }
        std::sort(SortedValues.begin(), SortedValues.end());

        double GL = 0.0, HL = 0.0;
        for (size_t i = 0; i < SortedValues.size() - 1; ++i) {
            GL += Gradients[SortedValues[i].second];
            HL += Hessians[SortedValues[i].second];
            double GR = G - GL;
            double HR = H - HL;

            if (HL >= MinLeafSize && HR >= MinLeafSize) {
                double Gain = GL * GL / (HL + Lambda) + 
                             GR * GR / (HR + Lambda) - 
                             CurrentScore - Gamma;

                if (Gain > BestGain && 
                    SortedValues[i].first != SortedValues[i + 1].first) {
                    BestGain = Gain;
                    BestFeature = feature;
                    BestThreshold = (SortedValues[i].first + 
                                   SortedValues[i + 1].first) / 2.0;
                }
            }
        }
    }

    return {BestFeature, BestThreshold, BestGain};
}

/**
 * @brief Calculate the weight of a leaf
 * @param Indices Indices of samples in the leaf
 * @param Gradients Gradients vector
 * @param Hessians Hessians vector
 * @return The calculated weight for this leaf
 */
double DecisionTreeXGBoost::calculateLeafWeight(const std::vector<int>& Indices,
                                                const std::vector<double>& Gradients,
                                                const std::vector<double>& Hessians) {
    double G = sumGradients(Gradients, Indices);
    double H = sumHessians(Hessians, Indices);
    return -G / (H + Lambda);
}

/**
 * @brief Calculate the sum of gradients for specified samples
 * @param Gradients Gradient vector
 * @param Indices Indices of samples
 * @return The sum of gradients
 */
double DecisionTreeXGBoost::sumGradients(const std::vector<double>& Gradients, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int idx : Indices) {
        Sum += Gradients[idx];
    }
    return Sum;
}

/**
 * @brief Calculate the sum of hessians for specified samples
 * @param Hessians Hessian vector
 * @param Indices Indices of samples
 * @return The sum of hessians
 */
double DecisionTreeXGBoost::sumHessians(const std::vector<double>& Hessians, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int idx : Indices) {
        Sum += Hessians[idx];
    }
    return Sum;
}

/**
 * @brief Save the tree to a file
 * @param filename Name of the file where the tree will be saved
 */
void DecisionTreeXGBoost::saveTree(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for saving the tree.");
    }
    serializeNode(Root.get(), outFile);
    outFile.close();
}

/**
 * @brief Load a tree from a file
 * @param filename Name of the file containing the saved tree
 */
void DecisionTreeXGBoost::loadTree(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for loading the tree.");
    }
    Root = deserializeNode(inFile);
    inFile.close();
}

/**
 * @brief Serialize a node for writing to a file
 * @param node Pointer to the node to serialize
 * @param out Output stream to write the node data
 */
void DecisionTreeXGBoost::serializeNode(const Tree* node, std::ostream& out) {
    if (!node) {
        out << "#\n";
        return;
    }
    out << node->FeatureIndex << " " << node->MaxValue << " " << node->Prediction << " " << node->IsLeaf << "\n";
    serializeNode(node->Left.get(), out);
    serializeNode(node->Right.get(), out);
}

/**
 * @brief Deserialize a node from an input stream
 * @param in Input stream containing the serialized data
 * @return A unique pointer to the deserialized node
 */
std::unique_ptr<DecisionTreeXGBoost::Tree> DecisionTreeXGBoost::deserializeNode(std::istream& in) {
    std::string line;
    if (!std::getline(in, line) || line == "#") return nullptr;

    auto node = std::make_unique<Tree>();
    std::istringstream ss(line);
    ss >> node->FeatureIndex >> node->MaxValue >> node->Prediction >> node->IsLeaf;

    node->Left = deserializeNode(in);
    node->Right = deserializeNode(in);

    return node;
}

std::map<int, double> DecisionTreeXGBoost::getFeatureImportance() const {
    std::map<int, double> importance;
    calculateFeatureImportanceRecursive(Root.get(), importance);
    return importance;
}

void DecisionTreeXGBoost::calculateFeatureImportanceRecursive(
    const Tree* node, std::map<int, double>& importance) const {
    if (!node || node->IsLeaf) {
        return;
    }

    // Add the gain improvement to the feature importance
    importance[node->FeatureIndex] += node->GainImprovement;

    // Recursively calculate importance for subtrees
    calculateFeatureImportanceRecursive(node->Left.get(), importance);
    calculateFeatureImportanceRecursive(node->Right.get(), importance);
}
