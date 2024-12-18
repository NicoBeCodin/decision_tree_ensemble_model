#include <gtest/gtest.h>
#include "../ensemble_boosting/boosting.h"
#include "../ensemble_boosting/loss_function.h"
#include <vector>
#include <memory>
#include <cmath>

class BoostingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Création d'un jeu de données simple pour les tests
        X = {
            {2.0, 3.0}, {1.0, 2.0}, {3.0, 4.0}, {4.0, 5.0}, {2.0, 5.0},
            {5.0, 1.0}, {6.0, 2.0}, {7.0, 3.0}, {4.0, 1.0}, {8.0, 2.0}
        };
        y = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0};
    }

    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

// Test de la construction du modèle de boosting
TEST_F(BoostingTest, Construction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Boosting model(5, 0.1, std::move(loss_function), 3, 2, 0.1);
    ASSERT_NO_THROW(model.train(X, y));
}

// Test des prédictions
TEST_F(BoostingTest, Prediction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Boosting model(5, 0.1, std::move(loss_function), 3, 2, 0.1);
    model.train(X, y);
    
    // Test avec un point d'entra��nement
    std::vector<double> sample = X[0];
    double prediction = model.predict(sample);
    EXPECT_NEAR(prediction, y[0], 1.0);

    // Test avec un nouveau point
    std::vector<double> new_sample = {2.5, 3.5};
    double new_prediction = model.predict(new_sample);
    EXPECT_GE(new_prediction, 1.0);
    EXPECT_LE(new_prediction, 3.0);
}

// Test de l'amélioration avec plus d'estimateurs
TEST_F(BoostingTest, EstimatorsComparison) {
    // Modèle avec peu d'estimateurs
    auto loss_function1 = std::make_unique<LeastSquaresLoss>();
    Boosting model_few(3, 0.1, std::move(loss_function1), 3, 2, 0.1);
    model_few.train(X, y);
    
    // Modèle avec plus d'estimateurs
    auto loss_function2 = std::make_unique<LeastSquaresLoss>();
    Boosting model_many(10, 0.1, std::move(loss_function2), 3, 2, 0.1);
    model_many.train(X, y);
    
    // Calculer les MSE
    double mse_few = 0.0;
    double mse_many = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred_few = model_few.predict(X[i]);
        double pred_many = model_many.predict(X[i]);
        mse_few += std::pow(pred_few - y[i], 2);
        mse_many += std::pow(pred_many - y[i], 2);
    }
    mse_few /= X.size();
    mse_many /= X.size();
    
    // Le modèle avec plus d'estimateurs devrait être meilleur
    EXPECT_LE(mse_many, mse_few);
}

// Test des paramètres du boosting
TEST_F(BoostingTest, Parameters) {
    // Test avec différents learning rates
    auto loss_function1 = std::make_unique<LeastSquaresLoss>();
    auto loss_function2 = std::make_unique<LeastSquaresLoss>();
    Boosting model_fast(5, 0.5, std::move(loss_function1), 3, 2, 0.1);
    Boosting model_slow(5, 0.1, std::move(loss_function2), 3, 2, 0.1);
    
    model_fast.train(X, y);
    model_slow.train(X, y);
    
    // Calculer les MSE
    double mse_fast = 0.0;
    double mse_slow = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred_fast = model_fast.predict(X[i]);
        double pred_slow = model_slow.predict(X[i]);
        mse_fast += std::pow(pred_fast - y[i], 2);
        mse_slow += std::pow(pred_slow - y[i], 2);
    }
    mse_fast /= X.size();
    mse_slow /= X.size();
    
    // Le modèle rapide converge plus vite sur ce petit jeu de données
    EXPECT_LE(mse_fast, mse_slow * 1.2);
}

// Test des cas limites
TEST_F(BoostingTest, EdgeCases) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Boosting model(5, 0.1, std::move(loss_function), 3, 2, 0.1);
    
    // Test avec un dataset vide
    std::vector<std::vector<double>> empty_X;
    std::vector<double> empty_y;
    EXPECT_NO_THROW(model.train(empty_X, empty_y));
    
    // Test avec un seul exemple
    std::vector<std::vector<double>> single_X = {{1.0, 1.0}};
    std::vector<double> single_y = {1.0};
    EXPECT_NO_THROW(model.train(single_X, single_y));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 