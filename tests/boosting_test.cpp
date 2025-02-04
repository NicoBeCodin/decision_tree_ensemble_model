#include <gtest/gtest.h>
#include "../src/ensemble/boosting/boosting.h"
#include "../src/ensemble/boosting/loss_function.h"
#include <vector>
#include <memory>
#include <cmath>

class BoostingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Création d'un jeu de données simple pour les tests
        X = {
            2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 2.0, 5.0,
            5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0, 8.0, 2.0
        };
        y = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0};
        rowLength = 2; // Chaque échantillon a 2 colonne
    }

    std::vector<double> X;
    std::vector<double> y;
    int rowLength;
};

// Test de la construction du modèle de boosting
TEST_F(BoostingTest, Construction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Boosting model(5, 0.1, std::move(loss_function), 3, 2, 0.1, 0, 0);
    ASSERT_NO_THROW(model.train(X, rowLength, y, 0));
}

// Test des prédictions
TEST_F(BoostingTest, Prediction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Boosting model(5, 0.1, std::move(loss_function), 3, 2, 0.1, 0, 0);
    model.train(X, rowLength, y, 0);
    
    // Test avec un point d'entraînement
    std::vector<double> sample = {X[0], X[1]};
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
    Boosting model_few(3, 0.1, std::move(loss_function1), 3, 2, 0.1, 0, 0);
    model_few.train(X, rowLength, y, 0);
    
    // Modèle avec plus d'estimateurs
    auto loss_function2 = std::make_unique<LeastSquaresLoss>();
    Boosting model_many(10, 0.1, std::move(loss_function2), 3, 2, 0.1, 0, 0);
    model_many.train(X, rowLength, y, 0);
    
    // Calculer les MSE
    double mse_few = 0.0;
    double mse_many = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> sample = {X[2 * i], X[2 * i + 1]};  // Extract 2D point from flattened array
        double pred_few = model_few.predict(sample);
        double pred_many = model_many.predict(sample);
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
    Boosting model_fast(5, 0.5, std::move(loss_function1), 3, 2, 0.1, 0, 0);
    Boosting model_slow(5, 0.1, std::move(loss_function2), 3, 2, 0.1, 0, 0);
    
    model_fast.train(X, rowLength, y, 0);
    model_slow.train(X, rowLength, y, 0);
    
    // Calculer les MSE
    double mse_fast = 0.0;
    double mse_slow = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> sample = {X[2 * i], X[2 * i + 1]};  // Extract 2D point from flattened array
        double pred_fast = model_fast.predict(sample);
        double pred_slow = model_slow.predict(sample);
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
    Boosting model(5, 0.1, std::move(loss_function), 3, 2, 0.1, 0, 0);
    
    // Test avec un dataset vide
    std::vector<double> empty_X;
    std::vector<double> empty_y;
    int zeroRowLength = 0;
    EXPECT_NO_THROW(model.train(empty_X, zeroRowLength, empty_y, 0));
    
    // Test avec un seul exemple
    std::vector<double> single_X = {1.0, 1.0};
    std::vector<double> single_y = {1.0};
    EXPECT_NO_THROW(model.train(single_X, rowLength, single_y,0));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
