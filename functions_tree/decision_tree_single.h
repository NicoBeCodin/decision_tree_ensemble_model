//decision_tree_single.h
//Yifan
//14.11
#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H
#include "splitting_criteria.h"

#include <vector>
#include <memory>
#include <tuple>

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
    void SplitNode(Tree* Node, const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth);
    
    /**
     * @brief Fonction pour trouver la meilleure caractéristique et le meilleur seuil pour la division
     */
    std::tuple<int, double, double> FindBestSplit(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, const std::vector<int>& Indices, double CurrentMSE);

    /**
     * @brief Calculer la moyenne des valeurs d'un nœud
     */
    double CalculateMean(const std::vector<double>& Labels, const std::vector<int>& Indices);

    /**
     * @brief Calculer l'erreur quadratique moyenne (MSE)
     */
    double CalculateMSE(const std::vector<double>& Labels, const std::vector<int>& Indices);
    
    /**
     * @brief Pré-trier les indices des caractéristiques
     */
    std::vector<std::vector<int>> PreSortFeatures(const std::vector<std::vector<double>>& Data, const std::vector<int>& Indices);

public:
    /**
     * @brief Constructeur : initialise la profondeur maximale, la taille minimale des feuilles, l'erreur minimale et le critère de division
     */
    DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError, SplittingCriteria* Criteria);
    
    /**
     * @brief Fonction pour entraîner l'arbre de décision
     */
    void Train(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels);
    
    /**
     * @brief Fonction pour prédire une valeur pour un échantillon donné
     */
    double Predict(const std::vector<double>& Sample) const;
};

#endif
