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
/*
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
            X.push_back({x1, x2});
            // Fonction non-linéaire avec bruit
            y.push_back(std::sin(x1) * std::cos(x2) + dis_noise(gen));
        }
    }

    // Fonction pour effectuer la validation croisée k-fold
    std::vector<double> performKFoldCV(int k, std::function<double(const std::vector<std::vector<double>>&,
                                                                  const std::vector<double>&,
                                                                  const std::vector<std::vector<double>>&,
                                                                  const std::vector<double>&)> trainAndEvaluate) {
        std::vector<double> scores;
        int fold_size = X.size() / k;
        
        for (int i = 0; i < k; ++i) {
            std::vector<std::vector<double>> X_train, X_val;
            std::vector<double> y_train, y_val;
            
            // Création des ensembles d'entraînement et de validation
            for (size_t j = 0; j < X.size(); ++j) {
                if (j >= i * fold_size && j < (i + 1) * fold_size) {
                    X_val.push_back(X[j]);
                    y_val.push_back(y[j]);
                } else {
                    X_train.push_back(X[j]);
                    y_train.push_back(y[j]);
                }
            }
            
            double score = trainAndEvaluate(X_train, y_train, X_val, y_val);
            scores.push_back(score);
        }
        
        return scores;
    }

    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

// Test de la stabilité de la validation croisée pour Bagging
TEST_F(CrossValidationTest, BaggingCrossValidationStability) {
    const int k_folds = 5;
    
    auto baggingTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                     const std::vector<double>& y_train,
                                     const std::vector<std::vector<double>>& X_val,
                                     const std::vector<double>& y_val) {
        Bagging model(10, 3, 2, 0.1);
        model.train(X_train, y_train);
        return model.evaluate(X_val, y_val);
    };
    
    std::vector<double> scores = performKFoldCV(k_folds, baggingTrainAndEvaluate);
    
    double mean_score = calculateMean(scores);
    double std_score = calculateStd(scores, mean_score);
    
    // Vérifier que l'écart-type des scores n'est pas trop élevé
    EXPECT_LE(std_score, 0.5);
    // Vérifier que les scores sont dans une plage raisonnable
    EXPECT_GE(mean_score, 0.0);
    EXPECT_LE(mean_score, 2.0);
}

// Test de la stabilité de la validation croisée pour Boosting
TEST_F(CrossValidationTest, BoostingCrossValidationStability) {
    const int k_folds = 5;
    
    auto boostingTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                      const std::vector<double>& y_train,
                                      const std::vector<std::vector<double>>& X_val,
                                      const std::vector<double>& y_val) {
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        Boosting model(10, 0.1, std::move(loss_function), 3, 2, 0.1);
        model.train(X_train, y_train, 0);
        return model.evaluate(X_val, y_val);
    };
    
    std::vector<double> scores = performKFoldCV(k_folds, boostingTrainAndEvaluate);
    
    double mean_score = calculateMean(scores);
    double std_score = calculateStd(scores, mean_score);
    
    EXPECT_LE(std_score, 0.5);
    EXPECT_GE(mean_score, 0.0);
    EXPECT_LE(mean_score, 2.0);
}

// Test de la stabilité de la validation croisée pour XGBoost
TEST_F(CrossValidationTest, XGBoostCrossValidationStability) {
    const int k_folds = 5;
    
    auto xgboostTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                     const std::vector<double>& y_train,
                                     const std::vector<std::vector<double>>& X_val,
                                     const std::vector<double>& y_val) {
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        XGBoost model(10, 3, 0.1, 1.0, 0.0, std::move(loss_function));
        model.train(X_train, y_train);
        return model.evaluate(X_val, y_val);
    };
    
    std::vector<double> scores = performKFoldCV(k_folds, xgboostTrainAndEvaluate);
    
    double mean_score = calculateMean(scores);
    double std_score = calculateStd(scores, mean_score);
    
    EXPECT_LE(std_score, 0.5);
    EXPECT_GE(mean_score, 0.0);
    EXPECT_LE(mean_score, 2.0);
}

// Test de comparaison des performances entre les modèles
TEST_F(CrossValidationTest, ModelComparison) {
    const int k_folds = 5;
    
    // Obtenir les scores pour chaque modèle
    auto baggingTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                     const std::vector<double>& y_train,
                                     const std::vector<std::vector<double>>& X_val,
                                     const std::vector<double>& y_val) {
        Bagging model(10, 3, 2, 0.1);
        model.train(X_train, y_train);
        return model.evaluate(X_val, y_val);
    };
    
    auto boostingTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                      const std::vector<double>& y_train,
                                      const std::vector<std::vector<double>>& X_val,
                                      const std::vector<double>& y_val) {
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        Boosting model(10, 0.1, std::move(loss_function), 3, 2, 0.1);
        model.train(X_train, y_train, 0);
        return model.evaluate(X_val, y_val);
    };
    
    auto xgboostTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                     const std::vector<double>& y_train,
                                     const std::vector<std::vector<double>>& X_val,
                                     const std::vector<double>& y_val) {
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        XGBoost model(10, 3, 0.1, 1.0, 0.0, std::move(loss_function));
        model.train(X_train, y_train);
        return model.evaluate(X_val, y_val);
    };
    
    std::vector<double> bagging_scores = performKFoldCV(k_folds, baggingTrainAndEvaluate);
    std::vector<double> boosting_scores = performKFoldCV(k_folds, boostingTrainAndEvaluate);
    std::vector<double> xgboost_scores = performKFoldCV(k_folds, xgboostTrainAndEvaluate);
    
    double bagging_mean = calculateMean(bagging_scores);
    double boosting_mean = calculateMean(boosting_scores);
    double xgboost_mean = calculateMean(xgboost_scores);
    
    // Vérifier que tous les modèles ont des performances raisonnables
    EXPECT_GE(bagging_mean, 0.0);
    EXPECT_GE(boosting_mean, 0.0);
    EXPECT_GE(xgboost_mean, 0.0);
    EXPECT_LE(bagging_mean, 2.0);
    EXPECT_LE(boosting_mean, 2.0);
    EXPECT_LE(xgboost_mean, 2.0);
}

// Test de l'impact de différentes tailles de folds
TEST_F(CrossValidationTest, FoldSizeImpact) {
    std::vector<int> k_values = {3, 5, 10};
    
    auto modelTrainAndEvaluate = [](const std::vector<std::vector<double>>& X_train,
                                   const std::vector<double>& y_train,
                                   const std::vector<std::vector<double>>& X_val,
                                   const std::vector<double>& y_val) {
        Bagging model(10, 3, 2, 0.1);
        model.train(X_train, y_train);
        return model.evaluate(X_val, y_val);
    };
    
    std::vector<double> mean_scores;
    std::vector<double> std_scores;
    
    for (int k : k_values) {
        std::vector<double> scores = performKFoldCV(k, modelTrainAndEvaluate);
        mean_scores.push_back(calculateMean(scores));
        std_scores.push_back(calculateStd(scores, mean_scores.back()));
    }
    
    // Vérifier que les scores moyens sont similaires pour différentes valeurs de k
    for (size_t i = 1; i < mean_scores.size(); ++i) {
        EXPECT_NEAR(mean_scores[i], mean_scores[0], 0.5);
    }
    
    // Vérifier que l'écart-type reste dans une plage raisonnable
    for (double std_score : std_scores) {
        EXPECT_LE(std_score, 0.5); // L'écart-type ne devrait pas être trop grand
        EXPECT_GE(std_score, 0.0); // L'écart-type doit être positif
    }
    
    // Vérifier que les écarts-types sont du même ordre de grandeur
    for (size_t i = 1; i < std_scores.size(); ++i) {
        EXPECT_NEAR(std_scores[i], std_scores[i-1], 0.2); // Tolérance de 0.2 pour les variations
    }
}

// Test de l'impact des hyperparamètres de Bagging
TEST_F(CrossValidationTest, BaggingHyperparameters) {
    const int k_folds = 5;
    std::vector<int> n_trees_values = {5, 10, 20};
    std::vector<int> max_depth_values = {2, 3, 5};
    
    // Test de l'impact du nombre d'arbres
    std::vector<double> n_trees_scores;
    for (int n_trees : n_trees_values) {
        auto baggingTrainAndEvaluate = [n_trees](const std::vector<std::vector<double>>& X_train,
                                                const std::vector<double>& y_train,
                                                const std::vector<std::vector<double>>& X_val,
                                                const std::vector<double>& y_val) {
            Bagging model(n_trees, 3, 2, 0.1);
            model.train(X_train, y_train);
            return model.evaluate(X_val, y_val);
        };
        
        std::vector<double> scores = performKFoldCV(k_folds, baggingTrainAndEvaluate);
        n_trees_scores.push_back(calculateMean(scores));
    }
    
    // Plus d'arbres devraient donner de meilleures performances ou au moins similaires
    for (size_t i = 1; i < n_trees_scores.size(); ++i) {
        EXPECT_LE(n_trees_scores[i], n_trees_scores[i-1] * 1.1); // Tolérance de 10%
    }
    
    // Test de l'impact de la profondeur maximale
    std::vector<double> depth_scores;
    for (int max_depth : max_depth_values) {
        auto baggingTrainAndEvaluate = [max_depth](const std::vector<std::vector<double>>& X_train,
                                                  const std::vector<double>& y_train,
                                                  const std::vector<std::vector<double>>& X_val,
                                                  const std::vector<double>& y_val) {
            Bagging model(10, max_depth, 2, 0.1);
            model.train(X_train, y_train);
            return model.evaluate(X_val, y_val);
        };
        
        std::vector<double> scores = performKFoldCV(k_folds, baggingTrainAndEvaluate);
        depth_scores.push_back(calculateMean(scores));
    }
    
    // Plus de profondeur devrait permettre de mieux capturer la complexité
    for (size_t i = 1; i < depth_scores.size(); ++i) {
        EXPECT_LE(depth_scores[i], depth_scores[i-1] * 1.15); // Tolérance de 15%
    }
}

// Test de l'impact des hyperparamètres de Boosting
TEST_F(CrossValidationTest, BoostingHyperparameters) {
    const int k_folds = 5;
    std::vector<double> learning_rates = {0.05, 0.1, 0.2};
    std::vector<int> n_estimators = {5, 10, 20};
    
    // Test de l'impact du learning rate
    std::vector<double> lr_scores;
    for (double lr : learning_rates) {
        auto boostingTrainAndEvaluate = [lr](const std::vector<std::vector<double>>& X_train,
                                            const std::vector<double>& y_train,
                                            const std::vector<std::vector<double>>& X_val,
                                            const std::vector<double>& y_val) {
            auto loss_function = std::make_unique<LeastSquaresLoss>();
            Boosting model(10, lr, std::move(loss_function), 3, 2, 0.1);
            model.train(X_train, y_train,0);
            return model.evaluate(X_val, y_val);
        };
        
        std::vector<double> scores = performKFoldCV(k_folds, boostingTrainAndEvaluate);
        lr_scores.push_back(calculateMean(scores));
    }
    
    // Vérifier que les learning rates donnent des résultats raisonnables
    for (double score : lr_scores) {
        EXPECT_GE(score, 0.0);
        EXPECT_LE(score, 2.0);
    }
    
    // Test de l'impact du nombre d'estimateurs
    std::vector<double> n_estimators_scores;
    for (int n : n_estimators) {
        auto boostingTrainAndEvaluate = [n](const std::vector<std::vector<double>>& X_train,
                                           const std::vector<double>& y_train,
                                           const std::vector<std::vector<double>>& X_val,
                                           const std::vector<double>& y_val) {
            auto loss_function = std::make_unique<LeastSquaresLoss>();
            Boosting model(n, 0.1, std::move(loss_function), 3, 2, 0.1);
            model.train(X_train, y_train,0);
            return model.evaluate(X_val, y_val);
        };
        
        std::vector<double> scores = performKFoldCV(k_folds, boostingTrainAndEvaluate);
        n_estimators_scores.push_back(calculateMean(scores));
    }
    
    // Plus d'estimateurs devraient donner de meilleures performances ou au moins similaires
    for (size_t i = 1; i < n_estimators_scores.size(); ++i) {
        EXPECT_LE(n_estimators_scores[i], n_estimators_scores[i-1] * 1.1); // Tolérance de 10%
    }
}

// Test de l'impact des hyperparamètres de XGBoost
TEST_F(CrossValidationTest, XGBoostHyperparameters) {
    const int k_folds = 5;
    std::vector<double> lambda_values = {0.1, 1.0, 5.0}; // Régularisation L2
    std::vector<double> learning_rates = {0.05, 0.1, 0.2};
    
    // Test de l'impact de la régularisation L2
    std::vector<double> lambda_scores;
    for (double lambda : lambda_values) {
        auto xgboostTrainAndEvaluate = [lambda](const std::vector<std::vector<double>>& X_train,
                                               const std::vector<double>& y_train,
                                               const std::vector<std::vector<double>>& X_val,
                                               const std::vector<double>& y_val) {
            auto loss_function = std::make_unique<LeastSquaresLoss>();
            XGBoost model(10, 3, 0.1, lambda, 0.0, std::move(loss_function));
            model.train(X_train, y_train);
            return model.evaluate(X_val, y_val);
        };
        
        std::vector<double> scores = performKFoldCV(k_folds, xgboostTrainAndEvaluate);
        lambda_scores.push_back(calculateMean(scores));
    }
    
    // Plus de régularisation devrait réduire le surapprentissage
    for (double score : lambda_scores) {
        EXPECT_GE(score, 0.0);
        EXPECT_LE(score, 2.0);
    }
    
    // Test de l'impact du learning rate
    std::vector<double> lr_scores;
    for (double lr : learning_rates) {
        auto xgboostTrainAndEvaluate = [lr](const std::vector<std::vector<double>>& X_train,
                                           const std::vector<double>& y_train,
                                           const std::vector<std::vector<double>>& X_val,
                                           const std::vector<double>& y_val) {
            auto loss_function = std::make_unique<LeastSquaresLoss>();
            XGBoost model(10, 3, lr, 1.0, 0.0, std::move(loss_function));
            model.train(X_train, y_train);
            return model.evaluate(X_val, y_val);
        };
        
        std::vector<double> scores = performKFoldCV(k_folds, xgboostTrainAndEvaluate);
        lr_scores.push_back(calculateMean(scores));
    }
    
    // Vérifier que les learning rates donnent des résultats raisonnables
    for (double score : lr_scores) {
        EXPECT_GE(score, 0.0);
        EXPECT_LE(score, 2.0);
    }
}

// Test de sérialisation pour Bagging
TEST_F(CrossValidationTest, BaggingSerialization) {
    // Création et entraînement du modèle original
    Bagging original_model(10, 3, 2, 0.1);
    original_model.train(X, y);
    
    // Sauvegarder le modèle
    std::string filename = "bagging_model_test.txt";
    EXPECT_NO_THROW(original_model.save(filename));
    
    // Créer un nouveau modèle et le charger
    Bagging loaded_model(10, 3, 2, 0.1);
    EXPECT_NO_THROW(loaded_model.load(filename));
    
    // Vérifier que les prédictions sont identiques
    for (const auto& sample : X) {
        double pred_original = original_model.predict(sample);
        double pred_loaded = loaded_model.predict(sample);
        EXPECT_NEAR(pred_original, pred_loaded, 1e-6)
            << "Prédiction différente - Original: " << pred_original 
            << ", Chargé: " << pred_loaded;
    }
    
    // Vérifier que les performances sont identiques
    double score_original = original_model.evaluate(X, y);
    double score_loaded = loaded_model.evaluate(X, y);
    EXPECT_NEAR(score_original, score_loaded, 1e-6)
        << "Score différent - Original: " << score_original 
        << ", Chargé: " << score_loaded;
    
    // Nettoyer le fichier
    std::remove(filename.c_str());
}

// Test de sérialisation pour Boosting
TEST_F(CrossValidationTest, BoostingSerialization) {
    // Création et entraînement du modèle original
    auto loss_function_original = std::make_unique<LeastSquaresLoss>();
    Boosting original_model(10, 0.1, std::move(loss_function_original), 3, 2, 0.1);
    original_model.train(X, y, 0);
    
    // Sauvegarder le modèle
    std::string filename = "boosting_model_test.txt";
    EXPECT_NO_THROW(original_model.save(filename));
    
    // Créer un nouveau modèle et le charger
    auto loss_function_loaded = std::make_unique<LeastSquaresLoss>();
    Boosting loaded_model(10, 0.1, std::move(loss_function_loaded), 3, 2, 0.1);
    EXPECT_NO_THROW(loaded_model.load(filename));
    
    // Vérifier que les prédictions sont identiques
    for (const auto& sample : X) {
        double pred_original = original_model.predict(sample);
        double pred_loaded = loaded_model.predict(sample);
        EXPECT_NEAR(pred_original, pred_loaded, 1e-6)
            << "Prédiction différente - Original: " << pred_original 
            << ", Chargé: " << pred_loaded;
    }
    
    // Vérifier que les performances sont identiques
    double score_original = original_model.evaluate(X, y);
    double score_loaded = loaded_model.evaluate(X, y);
    EXPECT_NEAR(score_original, score_loaded, 1e-6)
        << "Score différent - Original: " << score_original 
        << ", Chargé: " << score_loaded;
    
    // Nettoyer le fichier
    std::remove(filename.c_str());
}

// Test de sérialisation pour XGBoost
TEST_F(CrossValidationTest, XGBoostSerialization) {
    // Création et entraînement du modèle original
    auto loss_function_original = std::make_unique<LeastSquaresLoss>();
    XGBoost original_model(10, 3, 0.1, 1.0, 0.0, std::move(loss_function_original));
    original_model.train(X, y);
    
    // Sauvegarder le modèle
    std::string filename = "xgboost_model_test.txt";
    EXPECT_NO_THROW(original_model.save(filename));
    
    // Créer un nouveau modèle et le charger
    auto loss_function_loaded = std::make_unique<LeastSquaresLoss>();
    XGBoost loaded_model(10, 3, 0.1, 1.0, 0.0, std::move(loss_function_loaded));
    EXPECT_NO_THROW(loaded_model.load(filename));
    
    // Vérifier que les prédictions sont identiques
    for (const auto& sample : X) {
        double pred_original = original_model.predict(sample);
        double pred_loaded = loaded_model.predict(sample);
        EXPECT_NEAR(pred_original, pred_loaded, 1e-6)
            << "Prédiction différente - Original: " << pred_original 
            << ", Chargé: " << pred_loaded;
    }
    
    // Vérifier que les performances sont identiques
    double score_original = original_model.evaluate(X, y);
    double score_loaded = loaded_model.evaluate(X, y);
    EXPECT_NEAR(score_original, score_loaded, 1e-6)
        << "Score différent - Original: " << score_original 
        << ", Chargé: " << score_loaded;
    
    // Nettoyer le fichier
    std::remove(filename.c_str());
}

// Test des cas d'erreur de sérialisation
TEST_F(CrossValidationTest, SerializationErrorCases) {
    // Test avec un fichier inexistant
    Bagging model(10, 3, 2, 0.1);
    std::string nonexistent_file = "nonexistent_file.txt";
    EXPECT_THROW({
        try {
            model.load(nonexistent_file);
        } catch (const std::runtime_error& e) {
            EXPECT_STREQ(e.what(), "Cannot open file for reading: nonexistent_file.txt");
            throw;
        }
    }, std::runtime_error);
    
    // Test avec un fichier non valide
    std::string invalid_filename = "invalid_model.txt";
    {
        std::ofstream invalid_file(invalid_filename);
        invalid_file << "Invalid model data";
    }
    
    EXPECT_THROW({
        try {
            model.load(invalid_filename);
        } catch (const std::runtime_error& e) {
            EXPECT_STREQ(e.what(), "Invalid file format");
            throw;
        }
    }, std::runtime_error);
    
    // Nettoyer le fichier
    std::remove(invalid_filename.c_str());
    
    // Test de sauvegarde avec un chemin invalide
    std::string invalid_path = "/invalid/path/that/does/not/exist/model.txt";
    EXPECT_THROW({
        try {
            model.save(invalid_path);
        } catch (const std::runtime_error& e) {
            EXPECT_STREQ(e.what(), "Cannot open file for writing: /invalid/path/that/does/not/exist/model.txt");
            throw;
        }
    }, std::runtime_error);
}

TEST(SerializationTest, BaggingSerializationTest) {
    // Création des données
    std::vector<std::vector<double>> X = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::vector<double> y = {2, 4, 6, 8};

    // Entraînement du modèle original
    Bagging original_model(3, 2, 2, 0.1);
    original_model.train(X, y);

    // Prédictions avant la sauvegarde
    std::vector<double> original_predictions;
    for (const auto& x : X) {
        original_predictions.push_back(original_model.predict(x));
    }

    // Sauvegarde du modèle
    original_model.save("bagging_model.txt");

    // Chargement dans un nouveau modèle
    Bagging loaded_model(3, 2, 2, 0.1);
    loaded_model.load("bagging_model.txt");

    // Prédictions après le chargement
    std::vector<double> loaded_predictions;
    for (const auto& x : X) {
        loaded_predictions.push_back(loaded_model.predict(x));
    }

    // Vérification des prédictions individuelles avec une tolérance plus élevée
    for (size_t i = 0; i < X.size(); ++i) {
        EXPECT_NEAR(original_predictions[i], loaded_predictions[i], 1e-6)
            << "Prédiction différente à l'index " << i 
            << "\nOriginal: " << original_predictions[i]
            << "\nChargé: " << loaded_predictions[i];
    }

    // Nettoyage
    std::remove("bagging_model.txt");
}

TEST(SerializationTest, BoostingSerializationTest) {
    // Création des données
    std::vector<std::vector<double>> X = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::vector<double> y = {2, 4, 6, 8};

    // Entraînement du modèle original
    auto loss = std::make_unique<LeastSquaresLoss>();
    Boosting original_model(3, 0.1, std::move(loss), 2, 2, 0.1);
    original_model.train(X, y, 0);

    // Prédictions avant la sauvegarde
    std::vector<double> original_predictions;
    for (const auto& x : X) {
        original_predictions.push_back(original_model.predict(x));
    }
    double original_initial_pred = original_model.getInitialPrediction();

    // Sauvegarde du modèle
    original_model.save("boosting_model.txt");

    // Chargement dans un nouveau modèle
    auto new_loss = std::make_unique<LeastSquaresLoss>();
    Boosting loaded_model(3, 0.1, std::move(new_loss), 2, 2, 0.1);
    loaded_model.load("boosting_model.txt");

    // Prédictions après le chargement
    std::vector<double> loaded_predictions;
    for (const auto& x : X) {
        loaded_predictions.push_back(loaded_model.predict(x));
    }
    double loaded_initial_pred = loaded_model.getInitialPrediction();

    // Vérification de la prédiction initiale avec une tolérance plus élevée
    EXPECT_NEAR(original_initial_pred, loaded_initial_pred, 1e-6)
        << "Prédiction initiale différente"
        << "\nOriginal: " << original_initial_pred
        << "\nChargé: " << loaded_initial_pred;

    // Vérification des prédictions individuelles avec une tolérance plus élevée
    for (size_t i = 0; i < X.size(); ++i) {
        EXPECT_NEAR(original_predictions[i], loaded_predictions[i], 1e-6)
            << "Prédiction différente à l'index " << i
            << "\nOriginal: " << original_predictions[i]
            << "\nChargé: " << loaded_predictions[i];
    }

    // Nettoyage
    std::remove("boosting_model.txt");
}

TEST(SerializationTest, XGBoostSerializationTest) {
    // Création des données
    std::vector<std::vector<double>> X = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::vector<double> y = {2, 4, 6, 8};

    // Entraînement du modèle original
    auto loss = std::make_unique<LeastSquaresLoss>();
    XGBoost original_model(3, 2, 0.1, 0.1, 0.1, std::move(loss));
    original_model.train(X, y);

    // Prédictions avant la sauvegarde
    std::vector<double> original_predictions;
    for (const auto& x : X) {
        original_predictions.push_back(original_model.predict(x));
    }
    double original_initial_pred = original_model.getInitialPrediction();

    // Sauvegarde du modèle
    original_model.save("xgboost_model.txt");

    // Chargement dans un nouveau modèle
    auto new_loss = std::make_unique<LeastSquaresLoss>();
    XGBoost loaded_model(3, 2, 0.1, 0.1, 0.1, std::move(new_loss));
    loaded_model.load("xgboost_model.txt");

    // Prédictions après le chargement
    std::vector<double> loaded_predictions;
    for (const auto& x : X) {
        loaded_predictions.push_back(loaded_model.predict(x));
    }
    double loaded_initial_pred = loaded_model.getInitialPrediction();

    // Vérification de la prédiction initiale avec une tolérance plus élevée
    EXPECT_NEAR(original_initial_pred, loaded_initial_pred, 1e-6)
        << "Prédiction initiale différente"
        << "\nOriginal: " << original_initial_pred
        << "\nChargé: " << loaded_initial_pred;

    // Vérification des prédictions individuelles avec une tolérance plus élevée
    for (size_t i = 0; i < X.size(); ++i) {
        EXPECT_NEAR(original_predictions[i], loaded_predictions[i], 1e-6)
            << "Prédiction différente à l'index " << i 
            << "\nOriginal: " << original_predictions[i]
            << "\nChargé: " << loaded_predictions[i];
    }

    // Nettoyage
    std::remove("xgboost_model.txt");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
*/