#include <gtest/gtest.h>
#include "../src/ensemble/bagging/bagging.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

/*

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
            {2.0, 3.0}, {1.0, 2.0}, {3.0, 4.0}, {4.0, 5.0}, {2.0, 5.0},
            {5.0, 1.0}, {6.0, 2.0}, {7.0, 3.0}, {4.0, 1.0}, {8.0, 2.0}
        };
        y = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0};
    }

    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

// Test de la construction du modèle de bagging
TEST_F(BaggingTest, Construction) {
    Bagging model(5, 3, 2, 0.1); // 5 arbres, profondeur 3, min_samples 2, min_error 0.1
    ASSERT_NO_THROW(model.train(X, y));
}

// Test des prédictions
TEST_F(BaggingTest, Prediction) {
    Bagging model(5, 3, 2, 0.1);
    model.train(X, y);
    
    // Test avec un point d'entraînement
    std::vector<double> sample = X[0];
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
    // Modèle avec peu d'arbres
    Bagging model_few(3, 3, 2, 0.1);
    // Modèle avec plus d'arbres
    Bagging model_many(10, 3, 2, 0.1);
    
    model_few.train(X, y);
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
    
    // Les deux modèles devraient avoir une erreur raisonnable
    EXPECT_LE(mse_few, 2.0);
    EXPECT_LE(mse_many, 2.0);
}

// Test de la profondeur des arbres
TEST_F(BaggingTest, TreeDepth) {
    // Modèle avec arbres peu profonds
    Bagging model_shallow(5, 2, 2, 0.1);
    // Modèle avec arbres profonds
    Bagging model_deep(5, 5, 2, 0.1);
    
    model_shallow.train(X, y);
    model_deep.train(X, y);
    
    // Calculer les MSE
    double mse_shallow = 0.0;
    double mse_deep = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred_shallow = model_shallow.predict(X[i]);
        double pred_deep = model_deep.predict(X[i]);
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
    Bagging model(10, 3, 2, 0.1);
    model.train(X, y);
    
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
    Bagging model(5, 3, 2, 0.1);
    
    // Test avec un dataset vide
    std::vector<std::vector<double>> empty_X;
    std::vector<double> empty_y;
    EXPECT_NO_THROW(model.train(empty_X, empty_y));
    
    // Test avec un seul exemple
    std::vector<std::vector<double>> single_X = {{1.0, 1.0}};
    std::vector<double> single_y = {1.0};
    EXPECT_NO_THROW(model.train(single_X, single_y));
    
    // Test avec des paramètres limites
    Bagging extreme_model(1, 1, 1, 0.0);
    EXPECT_NO_THROW(extreme_model.train(X, y));
}

// Test de l'évaluation
TEST_F(BaggingTest, Evaluation) {
    Bagging model(5, 3, 2, 0.1);
    model.train(X, y);
    
    double mse = model.evaluate(X, y);
    EXPECT_GE(mse, 0.0); // MSE doit être positive
    
    // L'erreur sur les données d'entraînement devrait être raisonnable
    EXPECT_LE(mse, 2.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
*/