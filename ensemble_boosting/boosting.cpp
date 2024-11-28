#include "boosting.h"
#include <numeric>
#include <iostream>


/**
 * @brief Constructeur pour initialiser le modèle de Boosting
 * @param n_estimators Nombre de faibles apprenants (arbres de décision)
 * @param max_depth Profondeur maximale pour chaque arbre
 * @param learning_rate Taux d'apprentissage
 * @param criteria Critère de division
 * @param loss_function Fonction de perte (pour calculer le gradient et la perte)
 */
Boosting::Boosting(int n_estimators, double learning_rate,
                   std::unique_ptr<LossFunction> loss_function,
                   int max_depth, int min_samples_split, double min_impurity_decrease)
    : n_estimators(n_estimators),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      min_impurity_decrease(min_impurity_decrease),
      learning_rate(learning_rate),
      loss_function(std::move(loss_function)),
      initial_prediction(0.0) {}

/**
 * @brief Initialiser la prédiction initiale avec la moyenne des valeurs y
 * @param y Vecteur des étiquettes cibles
 */
void Boosting::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}


/**
 * @brief Entraîner le modèle de Boosting
 * @param X Matrice des caractéristiques
 * @param y Vecteur des étiquettes cibles
 */
void Boosting::train(const std::vector<std::vector<double>>& X,
                     const std::vector<double>& y) {
    size_t n_samples = y.size();
    initializePrediction(y);
    std::vector<double> y_pred(n_samples, initial_prediction); //

    for (int i = 0; i < n_estimators; ++i) {

        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

   
        auto tree = std::make_unique<DecisionTreeSingle>(max_depth, min_samples_split, min_impurity_decrease);
        tree->train(X, residuals);

        for (size_t j = 0; j < n_samples; ++j) {
            y_pred[j] += learning_rate * tree->predict(X[j]);
        }

        estimators.push_back(std::move(tree));

       
        double loss = loss_function->computeLoss(y, y_pred);
        std::cout << " iteration " << i + 1 << "，value loss: " << loss << std::endl;
    }
}

/**
 * @brief Prédire pour un seul échantillon
 * @param x Vecteur des caractéristiques d'un échantillon
 * @return Prédiction pour l'échantillon
 */
double Boosting::predict(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& tree : estimators) {
        y_pred += learning_rate * tree->predict(x);
    }
    return y_pred;
}


/**
 * @brief Prédire pour plusieurs échantillons
 * @param X Matrice des caractéristiques
 * @return Vecteur des prédictions pour chaque échantillon
 */
std::vector<double> Boosting::predict(const std::vector<std::vector<double>>& X) const {
    size_t n_samples = X.size();
    std::vector<double> y_pred(n_samples, initial_prediction);

 
    for (const auto& tree : estimators) {
        for (size_t i = 0; i < n_samples; ++i) {
            y_pred[i] += learning_rate * tree->predict(X[i]);
        }
    }
    return y_pred;
}

/**
 * @brief Évaluer la performance du modèle sur un ensemble de test
 * @param X_test Matrice des caractéristiques de test
 * @param y_test Vecteur des étiquettes cibles de test
 * @return Erreur quadratique moyenne (MSE)
 */
double Boosting::evaluate(const std::vector<std::vector<double>>& X_test, const std::vector<double>& y_test) const {
    std::vector<double> y_pred = predict(X_test);
    return loss_function->computeLoss(y_test, y_pred);
}
