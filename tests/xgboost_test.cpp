#include <gtest/gtest.h>
#include "../src/ensemble/boosting_XGBoost/boosting_XGBoost.h"
#include "../src/ensemble/boosting/loss_function.h"
#include <vector>
#include <memory>
#include <cmath>
/*
class XGBoostTest : public ::testing::Test {
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

// Test de la construction du modèle XGBoost
TEST_F(XGBoostTest, Construction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    XGBoost model(5, 3, 0.1, 1.0, 0.0, std::move(loss_function));
    ASSERT_NO_THROW(model.train(X, y));
}

// Test des prédictions
TEST_F(XGBoostTest, Prediction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    XGBoost model(5, 3, 0.1, 1.0, 0.0, std::move(loss_function));
    model.train(X, y);
    
    // Test avec un point d'entraînement
    std::vector<double> sample = X[0];
    double prediction = model.predict(sample);
    EXPECT_NEAR(prediction, y[0], 1.0);

    // Test avec un nouveau point
    std::vector<double> new_sample = {2.5, 3.5};
    double new_prediction = model.predict(new_sample);
    // Les prédictions de XGBoost peuvent être plus variables à cause de la régularisation
    EXPECT_GE(new_prediction, 0.5);
    EXPECT_LE(new_prediction, 3.5);
}

// Test de l'effet du nombre d'estimateurs
TEST_F(XGBoostTest, EstimatorsComparison) {
    auto loss_function1 = std::make_unique<LeastSquaresLoss>();
    auto loss_function2 = std::make_unique<LeastSquaresLoss>();
    
    XGBoost model_few(3, 3, 0.1, 1.0, 0.0, std::move(loss_function1));
    XGBoost model_many(10, 3, 0.1, 1.0, 0.0, std::move(loss_function2));
    
    model_few.train(X, y);
    model_many.train(X, y);
    
    // Calculer les erreurs de prédiction moyennes
    double avg_error_few = 0.0;
    double avg_error_many = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred_few = model_few.predict(X[i]);
        double pred_many = model_many.predict(X[i]);
        avg_error_few += std::abs(pred_few - y[i]);
        avg_error_many += std::abs(pred_many - y[i]);
    }
    avg_error_few /= X.size();
    avg_error_many /= X.size();
    
    // Les deux modèles devraient avoir une erreur raisonnable
    EXPECT_LE(avg_error_few, 2.0);
    EXPECT_LE(avg_error_many, 2.0);
}

// Test de l'effet de la régularisation L2 (lambda)
TEST_F(XGBoostTest, RegularizationEffect) {
    auto loss_function1 = std::make_unique<LeastSquaresLoss>();
    auto loss_function2 = std::make_unique<LeastSquaresLoss>();
    
    // Modèle avec faible régularisation
    XGBoost model_low_reg(5, 3, 0.1, 0.1, 0.0, std::move(loss_function1));
    // Modèle avec forte régularisation
    XGBoost model_high_reg(5, 3, 0.1, 10.0, 0.0, std::move(loss_function2));
    
    model_low_reg.train(X, y);
    model_high_reg.train(X, y);
    
    // Calculer les MSE sur l'ensemble d'entraînement
    double mse_low_reg = 0.0;
    double mse_high_reg = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred_low = model_low_reg.predict(X[i]);
        double pred_high = model_high_reg.predict(X[i]);
        mse_low_reg += std::pow(pred_low - y[i], 2);
        mse_high_reg += std::pow(pred_high - y[i], 2);
    }
    mse_low_reg /= X.size();
    mse_high_reg /= X.size();
    
    // Le modèle avec moins de régularisation devrait mieux fitter les données d'entraînement
    EXPECT_LE(mse_low_reg, mse_high_reg);
}

// Test des paramètres de learning rate
TEST_F(XGBoostTest, LearningRateEffect) {
    auto loss_function1 = std::make_unique<LeastSquaresLoss>();
    auto loss_function2 = std::make_unique<LeastSquaresLoss>();
    
    XGBoost model_fast(5, 3, 0.5, 1.0, 0.0, std::move(loss_function1));
    XGBoost model_slow(5, 3, 0.1, 1.0, 0.0, std::move(loss_function2));
    
    model_fast.train(X, y);
    model_slow.train(X, y);
    
    // Calculer les erreurs moyennes absolues
    double mae_fast = 0.0;
    double mae_slow = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred_fast = model_fast.predict(X[i]);
        double pred_slow = model_slow.predict(X[i]);
        mae_fast += std::abs(pred_fast - y[i]);
        mae_slow += std::abs(pred_slow - y[i]);
    }
    mae_fast /= X.size();
    mae_slow /= X.size();
    
    // Les deux modèles devraient avoir une erreur raisonnable
    EXPECT_LE(mae_fast, 2.0);
    EXPECT_LE(mae_slow, 2.0);
}

// Test des cas limites
TEST_F(XGBoostTest, EdgeCases) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    XGBoost model(5, 3, 0.1, 1.0, 0.0, std::move(loss_function));
    
    // Test avec un dataset vide
    std::vector<std::vector<double>> empty_X;
    std::vector<double> empty_y;
    EXPECT_NO_THROW(model.train(empty_X, empty_y));
    
    // Test avec un seul exemple
    std::vector<std::vector<double>> single_X = {{1.0, 1.0}};
    std::vector<double> single_y = {1.0};
    EXPECT_NO_THROW(model.train(single_X, single_y));
    
    // Test avec des paramètres extrêmes
    auto loss_function_extreme = std::make_unique<LeastSquaresLoss>();
    XGBoost extreme_model(1, 1, 0.01, 100.0, 100.0, std::move(loss_function_extreme));
    EXPECT_NO_THROW(extreme_model.train(X, y));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
*/