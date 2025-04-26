#include <gtest/gtest.h>
#include "../src/functions/tree/decision_tree_single.h"
#include <vector>
#include <cmath>
#include <string>

class DecisionTreeTest : public ::testing::Test {
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

// Test de la construction de l'arbre
TEST_F(DecisionTreeTest, TreeConstruction) {
    DecisionTreeSingle tree(3, 2, 0.1, 0); // profondeur max = 3, min samples = 2, min error = 0.1, critère = MSE
    ASSERT_NO_THROW(tree.train(X, rowLength, y, 0)); // Critère = MSE
}

// Test de prédiction
TEST_F(DecisionTreeTest, Prediction) {
    DecisionTreeSingle tree(3, 2, 0.1, 0);
    tree.train(X, rowLength, y, 0);

    // Test avec un point d'entraînement
    std::vector<double> sample = {2.0, 3.0}; // Premier échantillon des données
    double prediction = tree.predict(sample.data(), sample.size());
    EXPECT_NEAR(prediction, 1.0, 1.0); // On accepte une erreur de 1.0

    // Test avec un nouveau point
    std::vector<double> new_sample = {2.5, 3.5};
    double new_prediction = tree.predict(new_sample.data(), new_sample.size());
    EXPECT_GE(new_prediction, 1.0);
    EXPECT_LE(new_prediction, 3.0);
}

// Test des cas limites
TEST_F(DecisionTreeTest, EdgeCases) {
    DecisionTreeSingle tree(3, 2, 0.1, 0);
    
    // Test avec un dataset vide
    std::vector<double> empty_data;
    std::vector<double> empty_labels;
    EXPECT_NO_THROW(tree.train(empty_data, rowLength, empty_labels, 0));

    // Test avec un seul exemple
    std::vector<double> single_data = {1.0, 1.0};
    std::vector<double> single_label = {1.0};
    EXPECT_NO_THROW(tree.train(single_data, rowLength, single_label, 0));
    
    // Test avec des paramètres limites
    DecisionTreeSingle zero_depth_tree(0, 2, 0.1, 0);
    EXPECT_NO_THROW(zero_depth_tree.train(X, rowLength, y, 0));
    
    DecisionTreeSingle min_samples_one_tree(3, 1, 0.1, 0);
    EXPECT_NO_THROW(min_samples_one_tree.train(X, rowLength, y, 0));
}

// Test de sauvegarde et chargement
TEST_F(DecisionTreeTest, SaveAndLoad) {
    DecisionTreeSingle tree1(3, 2, 0.1, 0);
    tree1.train(X, rowLength, y, 0);
    
    // Sauvegarde
    std::string filename = "test_tree.txt";
    EXPECT_NO_THROW(tree1.saveTree(filename));
    
    // Chargement dans un nouvel arbre
    DecisionTreeSingle tree2(3, 2, 0.1, 0);
    EXPECT_NO_THROW(tree2.loadTree(filename));
    
    // Vérification que les prédictions sont identiques
    for (size_t i = 0; i < y.size(); ++i) {
        std::vector<double> sample(X.begin() + i * rowLength, X.begin() + (i + 1) * rowLength);
        EXPECT_DOUBLE_EQ(tree1.predict(sample.data(), sample.size()), 
                                       tree2.predict(sample.data(), sample.size()));
    }
}

// Test des paramètres de l'arbre
TEST_F(DecisionTreeTest, TreeParameters) {
    // Test avec différentes profondeurs
    DecisionTreeSingle shallow_tree(1, 2, 0.1, 0);
    DecisionTreeSingle deep_tree(10, 2, 0.1, 0);
    
    shallow_tree.train(X, rowLength, y, 0);
    deep_tree.train(X, rowLength, y, 0);
    
    // L'arbre profond devrait avoir une meilleure précision sur les données d'entraînement
    double shallow_mse = 0.0;
    double deep_mse = 0.0;
    
    for (size_t i = 0; i < y.size(); ++i) {
        std::vector<double> sample(X.begin() + i * rowLength, X.begin() + (i + 1) * rowLength);
        double shallow_pred = shallow_tree.predict(sample.data(), sample.size());
        double deep_pred = deep_tree.predict(sample.data(), sample.size());
        shallow_mse += std::pow(shallow_pred - y[i], 2);
        deep_mse += std::pow(deep_pred - y[i], 2);
    }
    shallow_mse /= y.size();
    deep_mse /= y.size();
    
    EXPECT_GE(shallow_mse, deep_mse); // Le MSE de l'arbre profond doit être inférieur
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
