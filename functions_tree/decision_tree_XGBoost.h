#ifndef DECISION_TREE_XGBOOST_H
#define DECISION_TREE_XGBOOST_H

#include <vector>
#include <memory>
#include <tuple>
#include <ostream>
#include <istream>
#include <filesystem>
#include <string>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <queue>
#include <map>

// DecisionTree class for XGBoost
class DecisionTreeXGBoost {
public:
    struct Tree {
        int FeatureIndex = -1;
        double MaxValue = 0.0;
        double Prediction = 0.0;
        bool IsLeaf = false;
        double GainImprovement = 0.0;  // Gain improvement for this node
        std::unique_ptr<Tree> Left = nullptr;
        std::unique_ptr<Tree> Right = nullptr;
    };

    DecisionTreeXGBoost(int MaxDepth, int MinLeafSize, double Lambda, double Gamma);

    // Train the tree
    void train(const std::vector<std::vector<double>>& Data, 
               const std::vector<double>& Labels,
               std::vector<double>& Predictions);

    // Predict for a single sample
    double predict(const std::vector<double>& Sample) const;

    // Save and load the tree
    void saveTree(const std::string& filename);
    void loadTree(const std::string& filename);

    // Calculate feature importance
    std::map<int, double> getFeatureImportance() const;

    // Nouvelle méthode pour l'importance des caractéristiques
    const Tree* getRoot() const { return Root.get(); }

private:
    int MaxDepth;       // Maximum depth of the tree
    int MinLeafSize;    // Minimum number of samples per leaf
    double Lambda;      // L2 regularization
    double Gamma;       // Minimum gain for a split
    std::unique_ptr<Tree> Root;

    // Internal methods
    void splitNode(Tree* Node, const std::vector<std::vector<double>>& Data, 
                   const std::vector<double>& Gradients,
                   const std::vector<double>& Hessians,
                   const std::vector<int>& Indices, int Depth);

    std::tuple<int, double, double> findBestSplit(
        const std::vector<std::vector<double>>& Data,
        const std::vector<double>& Gradients,
        const std::vector<double>& Hessians,
        const std::vector<int>& Indices);

    double calculateLeafWeight(const std::vector<int>& Indices,
                               const std::vector<double>& Gradients,
                               const std::vector<double>& Hessians);

    void computeGradientsAndHessians(const std::vector<double>& Labels,
                                     const std::vector<double>& Predictions,
                                     std::vector<double>& Gradients,
                                     std::vector<double>& Hessians);

    void serializeNode(const Tree* node, std::ostream& out);
    std::unique_ptr<Tree> deserializeNode(std::istream& in);

    double sumGradients(const std::vector<double>& Gradients, const std::vector<int>& Indices);
    double sumHessians(const std::vector<double>& Hessians, const std::vector<int>& Indices);

    // Recursive method to calculate feature importance
    void calculateFeatureImportanceRecursive(const Tree* node, std::map<int, double>& importance) const;
};

#endif // DECISION_TREE_XGBOOST_H
