#include <gtest/gtest.h>
#include "../src/ensemble/bagging/bagging.h"
#include "../src/ensemble/boosting/loss_function.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

// Fonction utilitaire pour calculer la variance
static double calculateVariance(const std::vector<double>& values) {
    double mean = 0.0;
    for (double value : values) {
        mean += value;
    }
    mean /= values.size();
    
    double variance = 0.0;
    for (double value : values) {
        variance += std::pow(value - mean, 2);
    }
    return variance / values.size();
}

class BaggingTest : public ::testing::Test {
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

    std::vector<double> X; // Donnée linéarisé
    std::vector<double> y; // Labels correspondants
    int rowLength; // Nombre de colonne par échantillon
};

// Test de la construction du modèle de bagging
TEST_F(BaggingTest, Construction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Bagging model(5, 3, 2, 0.1, std::move(loss_function), 0, 0); // 5 arbres, profondeur 3, min_samples 2, min_error 0.1
    ASSERT_NO_THROW(model.train(X, rowLength, y, 0)); // Avec le critère MSE
}

// Test des prédictions
TEST_F(BaggingTest, Prediction) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Bagging model(5, 3, 2, 0.1, std::move(loss_function), 0, 0);
    model.train(X, rowLength, y, 0);
    
    // Test avec un point d'entraînement
    std::vector<double> sample = {X[0], X[1]};
    double prediction = model.predict(sample);
    EXPECT_NEAR(prediction, y[0], 1.0);

    // Test avec un nouveau point
    std::vector<double> new_sample = {2.5, 3.5};
    double new_prediction = model.predict(new_sample);
    // Les prédictions devraient être dans la plage des valeurs d'entraînement
    EXPECT_GE(new_prediction, *std::min_element(y.begin(), y.end()));
    EXPECT_LE(new_prediction, *std::max_element(y.begin(), y.end()));
}

// Test de l'effet du nombre d'arbres
TEST_F(BaggingTest, NumberOfTrees) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    // Modèle avec peu d'arbres
    Bagging model_few(3, 3, 2, 0.1, std::move(loss_function), 0, 0);
    // Modèle avec plus d'arbres
    Bagging model_many(10, 3, 2, 0.1, std::move(loss_function), 0, 0);
    
    model_few.train(X, rowLength, y, 0);
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
    
    // Les deux modèles devraient avoir une erreur raisonnable
    EXPECT_LE(mse_few, 2.0);
    EXPECT_LE(mse_many, 2.0);
}

// Test de la profondeur des arbres
TEST_F(BaggingTest, TreeDepth) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    // Modèle avec arbres peu profonds
    Bagging model_shallow(5, 2, 2, 0.1, std::move(loss_function), 0, 0);
    // Modèle avec arbres profonds
    Bagging model_deep(5, 5, 2, 0.1);
    
    model_shallow.train(X, rowLength, y, 0);
    model_deep.train(X, rowLength, y, 0);
    
    // Calculer les MSE
    double mse_shallow = 0.0;
    double mse_deep = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> sample = {X[2 * i], X[2 * i + 1]};  // Extract 2D point from flattened array
        double pred_shallow = model_shallow.predict(sample);
        double pred_deep = model_deep.predict(sample);
        mse_shallow += std::pow(pred_shallow - y[i], 2);
        mse_deep += std::pow(pred_deep - y[i], 2);
    }
    mse_shallow /= X.size();
    mse_deep /= X.size();
    
    // Le modèle avec des arbres plus profonds devrait mieux fitter les données d'entraînement
    EXPECT_LE(mse_deep, mse_shallow);
}

// Test de la stabilité des prédictions
TEST_F(BaggingTest, PredictionStability) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Bagging model(10, 3, 2, 0.1, std::move(loss_function), 0, 0);
    model.train(X, rowLength, y, 0);
    
    // Faire plusieurs prédictions pour le même point
    std::vector<double> new_sample = {2.5, 3.5};
    std::vector<double> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(model.predict(new_sample));
    }
    
    // Les prédictions devraient être identiques (pas de randomisation lors de la prédiction)
    for (size_t i = 1; i < predictions.size(); ++i) {
        EXPECT_DOUBLE_EQ(predictions[i], predictions[0]);
    }
}

// Test des cas limites
TEST_F(BaggingTest, EdgeCases) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();

    Bagging model(5, 3, 2, 0.1, std::move(loss_function), 0, 0);
    
    // Test avec un dataset vide
    std::vector<double> empty_X;
    int zeroRowLenght = 0;
    std::vector<double> empty_y;
    EXPECT_NO_THROW(model.train(empty_X, zeroRowLenght, empty_y, 0));
    
    // Test avec un seul exemple
    std::vector<double> single_X = {1.0, 1.0};
    int singleRowLenght = 1;
    std::vector<double> single_y = {1.0};
    EXPECT_NO_THROW(model.train(single_X, singleRowLenght, single_y, 0));
    
    // Test avec des paramètres limites
    Bagging extreme_model(1, 1, 1, 0.0, std::move(loss_function), 0, 0);
    EXPECT_NO_THROW(extreme_model.train(X, rowLength, y, 0));
}

// Test de l'évaluation
TEST_F(BaggingTest, EvaluationMSE) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Bagging model(5, 3, 2, 0.1, std::move(loss_function), 0, 0);
    model.train(X, rowLength, y, 0);
    
    double mse = model.evaluate(X, rowLength, y);
    EXPECT_GE(mse, 0.0); // MSE doit être positive
    
    // L'erreur sur les données d'entraînement devrait être raisonnable
    EXPECT_LE(mse, 2.0);
}

// Test de l'évaluation
TEST_F(BaggingTest, EvaluationMAE) {
    auto loss_function = std::make_unique<MeanAbsoluteLoss>();
    Bagging model(5, 3, 2, 0.1, std::move(loss_function), 1, 1);
    model.train(X, rowLength, y, 1);
    
    double mae = model.evaluate(X, rowLength, y);
    EXPECT_GE(mae, 0.0); // MSE doit être positive
    
    // L'erreur sur les données d'entraînement devrait être raisonnable
    EXPECT_LE(mae, 2.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}