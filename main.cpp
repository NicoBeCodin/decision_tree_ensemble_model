#include "functions_io/functions_io.h"
#include "functions_tree/decision_tree_single.h"
#include "functions_tree/math_functions.h"
#include "functions_tree/feature_importance.h"
#include "functions_tree/tree_visualization.h"
#include "ensemble_bagging/bagging.h"
#include "ensemble_boosting/boosting.h"
#include "ensemble_boosting/loss_function.h"
#include "ensemble_boosting_XGBoost/boosting_XGBoost.h"
#include <iostream>
#include <chrono>
#include <iomanip>

void displayFeatureImportance(const std::vector<FeatureImportance::FeatureScore>& scores) {
    std::cout << "\nImportance des caractéristiques :\n";
    std::cout << std::setw(15) << "Caractéristique" << std::setw(15) << "Importance (%)\n";
    std::cout << std::string(30, '-') << "\n";
    
    for (const auto& score : scores) {
        std::cout << std::setw(15) << score.feature_name 
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << score.importance_score * 100.0 << "\n";
    }
    std::cout << std::endl;
}

int main()
{
    DataIO data_io;
    auto [X, y] = data_io.readCSV("../datasets/cleaned_data.csv");
    if (X.empty() || y.empty())
    {
        std::cerr << "Unable to open the data file, please check the path." << std::endl;
        return -1;
    }

    // Noms des caractéristiques
    std::vector<std::string> feature_names = {
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
        "matrix_size_x", "matrix_size_y"
    };

    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    std::cout << "Choose the method you want to use:\n";
    std::cout << "1: Simple Decision Tree\n";
    std::cout << "2: Bagging\n";
    std::cout << "3: Boosting\n";
    std::cout << "4: Boosting modèle XGBoost\n";
    int choice;
    std::cin >> choice;

    if (choice == 1)
    {
        std::cout<< "Which method do you want as a splitting criteria: MSE (0) (faster for the moment) or MAE (1) ?"<<std::endl;
        int criteria;
        std::cin>>criteria;
        std::cout << "Training a single decision tree, please wait...\n";
        int maxDepth = 60;
        int minSamplesSplit = 2;
        double minImpurityDecrease = 1e-12;

        DecisionTreeSingle single_tree(maxDepth, minSamplesSplit, minImpurityDecrease);

        auto train_start = std::chrono::high_resolution_clock::now();
        single_tree.train(X_train, y_train, criteria);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training time: " << train_duration.count() << " seconds\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = 0.0;
        size_t test_size = X_test.size();
        std::vector<double> y_pred;
        y_pred.reserve(test_size);
        for (const auto &X : X_test)
        {
            y_pred.push_back(single_tree.predict(X));
        }
        mse_value = Math::computeLoss(y_test, y_pred);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;

        std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
        std::cout << "Mean Squared Error (MSE): " << mse_value << "\n";

        // Calcul et affichage de l'importance des caractéristiques
        auto feature_importance = FeatureImportance::calculateTreeImportance(single_tree, feature_names);
        displayFeatureImportance(feature_importance);

        std::cout << "Would you like to save this tree? (0 = no, 1 = yes)\n";
        int answer = 0;
        std::cin >> answer;
        if (answer == 1)
        {
            std::cout << "Please type the name you want to give to the .txt file: \n";
            std::string filename;
            std::cin >> filename;
            std::cout << "Saving tree as: " << filename << std::endl;
            single_tree.saveTree(filename);
        }

        // Ajout de la visualisation
        std::cout << "Génération de la visualisation de l'arbre..." << std::endl;
        TreeVisualization::generateDotFile(single_tree, "single_tree", feature_names);
        std::cout << "Visualisation générée dans le dossier 'visualizations'" << std::endl;
    }
    else if (choice == 2)
    {
        std::cout << "Bagging process started, please wait...\n";
        int num_trees = 20;
        int max_depth = 60;
        int min_samples_split = 2;
        double min_impurity_decrease = 1e-6;

        Bagging bagging_model(num_trees, max_depth, min_samples_split, min_impurity_decrease);

        auto train_start = std::chrono::high_resolution_clock::now();
        bagging_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training time (Bagging): " << train_duration.count() << " seconds\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = bagging_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evaluation time (Bagging): " << eval_duration.count() << " seconds\n";

        std::cout << "Bagging Mean Squared Error (MSE): " << mse_value << "\n";

        // Calcul et affichage de l'importance des caractéristiques pour le bagging
        auto feature_importance = FeatureImportance::calculateBaggingImportance(bagging_model, feature_names);
        displayFeatureImportance(feature_importance);

        // Ajout de la visualisation
        std::cout << "Génération des visualisations des arbres..." << std::endl;
        TreeVisualization::generateEnsembleDotFiles(bagging_model.getTrees(), "bagging", feature_names);
        std::cout << "Visualisations générées dans le dossier 'visualizations'" << std::endl;
    }
    else if (choice == 3)
    {
        std::cout << "Boosting process started, please wait...\n";
        int n_estimators = 20;
        int max_depth = 60;
        int min_samples_split = 2;
        double min_impurity_decrease = 1e-6;
        double learning_rate = 0.1;

        auto loss_function = std::make_unique<LeastSquaresLoss>();
        Boosting boosting_model(n_estimators, learning_rate, std::move(loss_function),
                                max_depth, min_samples_split, min_impurity_decrease);

        auto train_start = std::chrono::high_resolution_clock::now();
        boosting_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training time: " << train_duration.count() << " seconds\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = boosting_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";

        std::cout << "Boosting Mean Squared Error (MSE): " << mse_value << "\n";

        // Calcul et affichage de l'importance des caractéristiques pour le boosting
        auto feature_importance = FeatureImportance::calculateBoostingImportance(boosting_model, feature_names);
        displayFeatureImportance(feature_importance);

        // Ajout de la visualisation
        std::cout << "Génération des visualisations des arbres..." << std::endl;
        TreeVisualization::generateEnsembleDotFiles(boosting_model.getEstimators(), "boosting", feature_names);
        std::cout << "Visualisations générées dans le dossier 'visualizations'" << std::endl;
    }
    else if (choice == 4)
    {
        std::cout << "Boosting process started, please wait...\n";
        int n_estimators = 20;
        int max_depth = 60;
        int min_samples_split = 2;
        double min_impurity_decrease = 1e-6;
        double learning_rate = 0.1;
        double lambda = 1.0;
        double alpha = 0.0;

        auto loss_function = std::make_unique<LeastSquaresLoss>();
        XGBoost XGBoost_model(n_estimators, max_depth, learning_rate, lambda, alpha, std::move(loss_function));

        auto train_start = std::chrono::high_resolution_clock::now();
        XGBoost_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training time: " << train_duration.count() << " seconds\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = XGBoost_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";

        std::cout << "Boosting Mean Squared Error (MSE): " << mse_value << "\n";
    }
    else
    {
        std::cerr << "Invalid choice! Please rerun the program and choose 1, 2, or 3." << std::endl;
        return -1;
    }

    return 0;
}
