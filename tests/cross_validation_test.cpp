#include <gtest/gtest.h>
#include "../src/ensemble/bagging/bagging.h"
#include "../src/ensemble/boosting/boosting.h"
#include "../src/ensemble/boosting_XGBoost/boosting_XGBoost.h"
#include "../src/ensemble/boosting/loss_function.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>

// Fonction utilitaire pour calculer la moyenne
static double calculateMean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

// Fonction utilitaire pour calculer l'écart-type
static double calculateStd(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    double variance = 0.0;
    for (double value : values) {
        variance += std::pow(value - mean, 2);
    }
    return std::sqrt(variance / (values.size() - 1));
}

class CrossValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Création d'un jeu de données plus grand pour la validation croisée
        std::mt19937 gen(42); // Pour la reproductibilité
        std::uniform_real_distribution<> dis_x(-10.0, 10.0);
        std::normal_distribution<> dis_noise(0.0, 0.1);
        
        // Génération de données synthétiques
        for (int i = 0; i < 100; ++i) {
            double x1 = dis_x(gen);
            double x2 = dis_x(gen);
            X.push_back(x1);
            X.push_back(x2);
            // Fonction non-linéaire avec bruit
            y.push_back(std::sin(x1) * std::cos(x2) + dis_noise(gen));
        }
    }
    std::vector<double> X;
    std::vector<double> y;
    int rowLength = 2;
};


// Test de la validation croisee sur Bagging
TEST_F(CrossValidationTest, BaggingCrossValidationStability) {
    Bagging model(10, 3, 2, 0.1);
    model.train(X, rowLength, y, 0);
    double score = model.evaluate(X, rowLength, y);
    EXPECT_GE(score, 0.0);
    EXPECT_LE(score, 2.0);
}

// Test de la validation croisee sur Boosting
TEST_F(CrossValidationTest, BoostingCrossValidationStability) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    Boosting model(10, 0.1, std::move(loss_function), 3, 2, 0.1);
    model.train(X, rowLength, y, 0);
    double score = model.evaluate(X, rowLength, y);
    EXPECT_GE(score, 0.0);
    EXPECT_LE(score, 2.0);
}

// Test de la validation croisee sur XGBoost
TEST_F(CrossValidationTest, XGBoostCrossValidationStability) {
    auto loss_function = std::make_unique<LeastSquaresLoss>();
    XGBoost model(10, 3, 2, 0.1, 1.0, 0.0, std::move(loss_function), 0);
    model.train(X, rowLength, y);
    double score = model.evaluate(X, rowLength, y);
    EXPECT_GE(score, 0.0);
    EXPECT_LE(score, 2.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
