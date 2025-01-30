#include <gtest/gtest.h>
#include "../src/functions/tree/decision_tree_single.h"
#include <vector>
#include <cmath>

/*
class DecisionTreeTest : public ::testing::Test {
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

// Test de la construction de l'arbre
TEST_F(DecisionTreeTest, TreeConstruction) {
    DecisionTreeSingle tree(3, 2, 0.1); // profondeur max = 3, min samples = 2, min error = 0.1
    ASSERT_NO_THROW(tree.train(X, y, 0)); // 0 pour MSE comme critère
}

// Test de prédiction
TEST_F(DecisionTreeTest, Prediction) {
    DecisionTreeSingle tree(3, 2, 0.1);
    tree.train(X, y, 0);
    
    // Test avec un point d'entraînement
    std::vector<double> sample = X[0];
    double prediction = tree.predict(sample);
    EXPECT_NEAR(prediction, y[0], 1.0); // On accepte une erreur de 1.0

    // Test avec un nouveau point
    std::vector<double> new_sample = {2.5, 3.5};
    double new_prediction = tree.predict(new_sample);
    EXPECT_GE(new_prediction, 1.0);
    EXPECT_LE(new_prediction, 3.0);
}

// Test des cas limites
TEST_F(DecisionTreeTest, EdgeCases) {
    DecisionTreeSingle tree(3, 2, 0.1);
    
    // Test avec un dataset vide
    std::vector<std::vector<double>> empty_X;
    std::vector<double> empty_y;
    EXPECT_NO_THROW(tree.train(empty_X, empty_y, 0));

    // Test avec un seul exemple
    std::vector<std::vector<double>> single_X = {{1.0, 1.0}};
    std::vector<double> single_y = {1.0};
    EXPECT_NO_THROW(tree.train(single_X, single_y, 0));
    
    // Test avec des paramètres limites
    DecisionTreeSingle zero_depth_tree(0, 2, 0.1);
    EXPECT_NO_THROW(zero_depth_tree.train(X, y, 0));
    
    DecisionTreeSingle min_samples_one_tree(3, 1, 0.1);
    EXPECT_NO_THROW(min_samples_one_tree.train(X, y, 0));
}

// Test de sauvegarde et chargement
TEST_F(DecisionTreeTest, SaveAndLoad) {
    DecisionTreeSingle tree1(3, 2, 0.1);
    tree1.train(X, y, 0);
    
    // Sauvegarde
    std::string filename = "test_tree.txt";
    EXPECT_NO_THROW(tree1.saveTree(filename));
    
    // Chargement dans un nouvel arbre
    DecisionTreeSingle tree2(3, 2, 0.1);
    EXPECT_NO_THROW(tree2.loadTree(filename));
    
    // Vérification que les prédictions sont identiques
    for (const auto& sample : X) {
        EXPECT_DOUBLE_EQ(tree1.predict(sample), tree2.predict(sample));
    }
}

// Test des paramètres de l'arbre
TEST_F(DecisionTreeTest, TreeParameters) {
    // Test avec différentes profondeurs
    DecisionTreeSingle shallow_tree(1, 2, 0.1);
    DecisionTreeSingle deep_tree(10, 2, 0.1);
    
    shallow_tree.train(X, y, 0);
    deep_tree.train(X, y, 0);
    
    // L'arbre profond devrait avoir une meilleure précision sur les données d'entraînement
    double shallow_mse = 0.0;
    double deep_mse = 0.0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        double shallow_pred = shallow_tree.predict(X[i]);
        double deep_pred = deep_tree.predict(X[i]);
        shallow_mse += std::pow(shallow_pred - y[i], 2);
        deep_mse += std::pow(deep_pred - y[i], 2);
    }
    shallow_mse /= X.size();
    deep_mse /= X.size();
    
    EXPECT_GE(shallow_mse, deep_mse);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 

*/