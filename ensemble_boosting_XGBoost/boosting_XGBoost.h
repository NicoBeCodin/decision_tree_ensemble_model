#ifndef BOOSTING_XGBOOST_H
#define BOOSTING_XGBOOST_H

#include "../functions_tree/decision_tree_XGBoost.h"
#include "../ensemble_boosting/loss_function.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <map>

/**
 * @brief Classe principale implémentant XGBoost.
 */
class XGBoost {
private:
    int n_estimators;
    int max_depth;
    double learning_rate;
    double lambda;  // L2 regularization
    double alpha;   // L1 regularization
    std::unique_ptr<LossFunction> loss_function;
    std::vector<std::unique_ptr<DecisionTreeXGBoost>> trees;
    double initial_prediction;

    /**
     * @brief Initialisation de la prédiction initiale avec la moyenne des valeurs y.
     * @param y Vecteur des étiquettes cibles)
     */
    void initializePrediction(const std::vector<double>& y);

public:
    /**
     * @brief Constructeur pour initialiser le modèle XGBoost pour le boosting
     * @param n_estimators Nombre de faibles apprenants (arbres de décision)
     * @param max_depth Profondeur maximale pour chaque arbre
     * @param learning_rate Taux d'apprentissage
     * @param lambda Paramètre de régularisation L2
     * @param alpha Paramètre de régularisation L1
     * @param loss_function Fonction de perte (pour calculer le gradient et la perte)
     */
    XGBoost(int n_estimators, int max_depth, double learning_rate, double lambda, double alpha, std::unique_ptr<LossFunction> loss_function);
    
    /**
     * @brief Entraîner le modèle de Boosting
     * @param X Matrice des caractéristiques
     * @param y Vecteur des étiquettes cibles
     */
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    
    /**
     * @brief Prédire pour un seul échantillon
     * @param x Vecteur des caractéristiques d'un échantillon
     * @return Prédiction pour l'échantillon
     */
    double predict(const std::vector<double>& x) const;

    /**
     * @brief Prédire pour plusieurs échantillons
     * @param X Matrice des caractéristiques
     * @return Vecteur des prédictions pour chaque échantillon
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    
    /**
     * @brief Évaluer la performance du modèle sur un ensemble de test
     * @param X_test Matrice des caractéristiques de test
     * @param y_test Vecteur des étiquettes cibles de test
     * @return Erreur quadratique moyenne (MSE)
     */
    double evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const;

    /**
     * @brief Calculer l'importance des caractéristiques
     * @param feature_names Noms des caractéristiques (optionnel)
     * @return Map des caractéristiques avec leur importance relative
     */
    std::map<std::string, double> featureImportance(const std::vector<std::string>& feature_names = {}) const;
    
    /** 
     * @brief Destructeur de XGBoost
    */
    ~XGBoost() = default;

    // Méthodes de sérialisation
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    double getInitialPrediction() const { return initial_prediction; }
};

#endif // BOOSTING_XGBOOST_H
