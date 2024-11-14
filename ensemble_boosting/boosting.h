//boosting
//Yifan
//14.11
#ifndef BOOSTING_H

#define BOOSTING_H

#include "../functions_tree/regression_tree.h"
#include "loss_function.h"
#include <vector>
#include <memory>

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
    Boosting(int n_estimators, int max_depth, double learning_rate,
             SplittingCriteria* criteria, std::unique_ptr<LossFunction> loss_function);

    /**
     * @brief Entraîner le modèle de boosting
     * @param X Matrice des caractéristiques d'entraînement
     * @param y Vecteur des étiquettes cibles d'entraînement
     */
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& y);

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


private:
    int n_estimators;   // Nombre de faibles apprenants (arbres de décision)
    int max_depth;      // Profondeur maximale pour chaque arbre
    double learning_rate; // Taux d'apprentissage
    SplittingCriteria* criteria;  // Critère de division pour les arbres
    std::unique_ptr<LossFunction> loss_function; // Fonction de perte à minimiser

    std::vector<std::unique_ptr<RegressionTree>> estimators; // Collection de faibles apprenants
    double initial_prediction; // Prédiction initiale (modèle constant)

    /**
     * @brief Initialiser la prédiction initiale basée sur le vecteur cible
     * @param y Vecteur des étiquettes cibles
     */
    void initializePrediction(const std::vector<double>& y);
};

#endif // BOOSTING_H
