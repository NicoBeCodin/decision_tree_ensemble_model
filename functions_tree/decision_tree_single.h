//decision_tree_single.h
//Yifan
//14.11
#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <memory>
#include <tuple>
#include "splitting_criteria.h"


class DecisionTreeSingle
{
private:
    struct Tree
    {
        int FeatureIndex = -1; // Indice de la caractéristique pour la division
        double MaxValue = 0.0; // Seuil pour la division
        double Prediction = 0.0; // Valeur de prédiction pour le nœud
        bool IsLeaf = false; // Indicateur pour vérifier si c'est une feuille
        std::unique_ptr<Tree> Left = nullptr;  // Pointeur intelligent vers le nœud gauche
        std::unique_ptr<Tree> Right = nullptr; // Pointeur intelligent vers le nœud droit
    };
    std::unique_ptr<Tree> Root; // Pointeur intelligent vers le nœud racine
    int MaxDepth;               // Profondeur maximale de l'arbre
    int MinLeafLarge;           // Taille minimale des feuilles
    double MinError;            // Erreur minimale pour la division
    SplittingCriteria* Criteria; // Pointeur vers le critère de division

    /**
     * @brief Fonction pour diviser un nœud
     */
    void splitNode(Tree* Node, const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth);
    
    /**
     * @brief Fonction pour trouver la meilleure caractéristique et le meilleur seuil pour la division
     */
    std::tuple<int, double, double> findBestSplit(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, const std::vector<int>& Indices, double CurrentMSE);

    /**
     * @brief Calculer la moyenne des valeurs d'un nœud
     */
    double calculateMean(const std::vector<double>& Labels, const std::vector<int>& Indices);

    /**
     * @brief Calculer l'erreur quadratique moyenne (MSE)
     */
    double calculateMSE(const std::vector<double>& Labels, const std::vector<int>& Indices);
    
    /**
     * @brief Pré-trier les indices des caractéristiques
     */
    std::vector<std::vector<int>> preSortFeatures(const std::vector<std::vector<double>>& Data, const std::vector<int>& Indices);

    


    /**
     * @brief Recursive function to serialize a node
     */
    void serializeNode(const Tree* node, std::ostream& out);

    /**
     * 
     * @brief Recursive function to deserialize a node
     */
    std::unique_ptr<DecisionTreeSingle::Tree> deserializeNode(std::istream& in);

public:
    /**
     * @brief Constructeur : initialise la profondeur maximale, la taille minimale des feuilles, l'erreur minimale et le critère de division
     */
    DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError, SplittingCriteria* Criteria);
    
    /**
     * @brief Fonction pour entraîner l'arbre de décision
     */
    void train(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels);
    
    /**
     * @brief Fonction pour prédire une valeur pour un échantillon donné
     */
    double predict(const std::vector<double>& Sample) const;

    /** 
    * @brief Save the tree to a file
    */
   void saveTree(const std::string &filename);

    /** 
    * @brief Load the tree from a file
    */
   void loadTree(const std::string &filename);


};

#endif
