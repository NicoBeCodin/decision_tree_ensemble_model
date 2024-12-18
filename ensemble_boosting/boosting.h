#ifndef BOOSTING_H
#define BOOSTING_H

#include "../functions_tree/decision_tree_single.h"
#include "loss_function.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>

/**
 * @brief Boosting 类
 * 实现梯度提升算法，使用多个弱学习器（决策树）
 */
class Boosting {
public:
     /**
     * @brief Constructeur pour initialiser le modèle de boosting
     * @param n_estimators Nombre de faibles apprenants (arbres de décision)
     * @param max_depth Profondeur maximale pour chaque arbre
     * @param learning_rate Taux d'apprentissage pour le modèle
     * @param criteria Critère de division pour les arbres
     * @param loss_function Fonction de perte à minimiser
     */
    Boosting(int n_estimators, double learning_rate,
             std::unique_ptr<LossFunction> loss_function,
             int max_depth, int min_samples_split, double min_impurity_decrease);

    /**
     * @brief Entraîner le modèle de boosting
     * @param X Matrice des caractéristiques d'entraînement
     * @param y Vecteur des étiquettes cibles d'entraînement
     */
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& y,
               const int criteria);

    /**
     * @brief Prédire une valeur pour un échantillon donné
     * @param x Vecteur représentant un échantillon
     * @return Prédiction pour l'échantillon
     */
    double predict(const std::vector<double>& x) const;

    /**
     * @brief Prédire des valeurs pour un ensemble d'échantillons
     * @param X Matrice représentant plusieurs échantillons
     * @return Vecteur des prédictions pour chaque échantillon
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    /**
     * @brief Évaluer le modèle sur un ensemble de test
     * @param X_test Matrice des caractéristiques de test
     * @param y_test Vecteur des étiquettes cibles de test
     * @return Erreur moyenne des prédictions sur l'ensemble de test
     */
    double evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const;

    // Nouvelle méthode pour accéder aux estimateurs
    const std::vector<std::unique_ptr<DecisionTreeSingle>>& getEstimators() const { return trees; }

    // Méthodes de sérialisation
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    double getInitialPrediction() const { return initial_prediction; }

private:
    int n_estimators;  
    int max_depth;      
    int min_samples_split;
    double min_impurity_decrease;
    double learning_rate; 
    std::unique_ptr<LossFunction> loss_function;

    std::vector<std::unique_ptr<DecisionTreeSingle>> trees; // 弱学习器集合
    double initial_prediction; 

    /**
     * @brief Initialiser la prédiction initiale basée sur le vecteur cible
     * @param y Vecteur des étiquettes cibles
     */
    void initializePrediction(const std::vector<double>& y);

    friend class FeatureImportance;
};

#endif // BOOSTING_H
