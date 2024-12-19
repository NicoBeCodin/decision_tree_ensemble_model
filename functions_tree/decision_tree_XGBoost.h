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

// Classe DecisionTree pour XGBoost
class DecisionTreeXGBoost {
public:
    struct Tree {
        int FeatureIndex = -1;
        double MaxValue = 0.0;
        double Prediction = 0.0;
        bool IsLeaf = false;
        double GainImprovement = 0.0;  // Amélioration du gain pour ce nœud
        std::unique_ptr<Tree> Left = nullptr;
        std::unique_ptr<Tree> Right = nullptr;
    };

    DecisionTreeXGBoost(int MaxDepth, int MinLeafSize, double Lambda, double Gamma);

    // Entraînement de l'arbre
    void train(const std::vector<std::vector<double>>& Data, 
               const std::vector<double>& Labels,
               std::vector<double>& Predictions);

    // Prédiction pour un échantillon unique
    double predict(const std::vector<double>& Sample) const;

    // Sauvegarder et charger l'arbre
    void saveTree(const std::string& filename);
    void loadTree(const std::string& filename);

    // Calculer l'importance des caractéristiques
    std::map<int, double> getFeatureImportance() const;

private:
    int MaxDepth;       // Profondeur maximale de l'arbre
    int MinLeafSize;    // Nombre minimal d'échantillons par feuille
    double Lambda;      // Régularisation L2
    double Gamma;       // Gain minimal pour un split
    std::unique_ptr<Tree> Root;

    // Méthodes internes
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

    // Méthode récursive pour calculer l'importance des caractéristiques
    void calculateFeatureImportanceRecursive(const Tree* node, std::map<int, double>& importance) const;
};

#endif // DECISION_TREE_XGBOOST_H
