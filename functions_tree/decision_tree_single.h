#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>
#include "math_functions.h"

class DecisionTreeSingle {
public:
    struct Tree {
        int FeatureIndex = -1;                 
        double MaxValue = 0.0;                 
        double Prediction = 0.0;               
        bool IsLeaf = false;                   
        double NodeMSE = 0.0;                  // MSE au niveau du nœud
        size_t NodeSamples = 0;               // Nombre d'échantillons au niveau du nœud
        std::unique_ptr<Tree> Left = nullptr;  
        std::unique_ptr<Tree> Right = nullptr; 
    };

    // Constructeur et méthodes existantes...
    DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError);
    void train(const std::vector<std::vector<double>> &Data, const std::vector<double> &Labels, int criteria);
    double predict(const std::vector<double> &Sample) const;
    void saveTree(const std::string &filename);
    void loadTree(const std::string &filename);

    // Nouvelles méthodes pour l'importance des caractéristiques
    const Tree* getRoot() const { return Root.get(); }
    double getRootMSE() const { return Root ? Root->NodeMSE : 0.0; }
    size_t getRootSamples() const { return Root ? Root->NodeSamples : 0; }

private:
    std::unique_ptr<Tree> Root;
    int MaxDepth;              
    int MinLeafLarge;          
    double MinError;           

    void splitNode(Tree *Node, const std::vector<std::vector<double>> &Data,
                  const std::vector<double> &Labels,
                  const std::vector<int> &Indices, int Depth);
    void splitNodeMAE(Tree *Node, const std::vector<std::vector<double>> &Data,
                  const std::vector<double> &Labels,
                  const std::vector<int> &Indices, int Depth);

    std::tuple<int, double, double>
    findBestSplit(const std::vector<std::vector<double>> &Data,
                  const std::vector<double> &Labels,
                  const std::vector<int> &Indices, double CurrentMSE);

    std::tuple<int, double, double>
    findBestSplitUsingMAE(const std::vector<std::vector<double>> &Data,
                          const std::vector<double> &Labels,
                          const std::vector<int> &Indices, double CurrentMAE);

    std::vector<std::vector<int>>
    preSortFeatures(const std::vector<std::vector<double>> &Data,
                    const std::vector<int> &Indices);

    void serializeNode(const Tree *node, std::ostream &out);
    std::unique_ptr<Tree> deserializeNode(std::istream &in);

    friend class FeatureImportance;
};

#endif // DECISION_TREE_SINGLE_H
