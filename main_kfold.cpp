#include "functions_io/functions_io.h"
#include "functions_tree/decision_tree_single.h"
#include "functions_tree/math_functions.h"
#include "ensemble_bagging/bagging.h"
#include "ensemble_boosting/boosting.h"
#include "ensemble_boosting/loss_function.h"
#include <iostream>
#include <chrono>
#include <memory>
#include <numeric>
#include <algorithm>
#include <random>


std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>> createKFolds(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& y, 
    int k)
{
    
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t fold_size = X.size() / k;
    std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>> folds;

    for (int i = 0; i < k; ++i)
    {
        std::vector<std::vector<double>> X_fold;
        std::vector<double> y_fold;

        size_t start = i * fold_size;
        size_t end = (i == k - 1) ? X.size() : (i + 1) * fold_size;

        for (size_t j = start; j < end; ++j)
        {
            X_fold.push_back(X[indices[j]]);
            y_fold.push_back(y[indices[j]]);
        }
        folds.emplace_back(X_fold, y_fold);
    }
    return folds;
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

    std::cout << "Choose the method you want to use:\n";
    std::cout << "1: Simple Decision Tree\n";
    std::cout << "2: Bagging\n";
    std::cout << "3: Boosting\n";
    int choice;
    std::cin >> choice;

    
    std::cout << "Please enter the number of folds for k-fold cross-validation (e.g., k=5): ";
    int k;
    std::cin >> k;

    
    auto folds = createKFolds(X, y, k);

    std::vector<double> mse_scores; 

    for (int i = 0; i < k; ++i)
    {
        
        std::vector<std::vector<double>> X_train, X_test;
        std::vector<double> y_train, y_test;

        for (int j = 0; j < k; ++j)
        {
            if (j == i)
            {
                
                X_test = folds[j].first;
                y_test = folds[j].second;
            }
            else
            {
                
                X_train.insert(X_train.end(), folds[j].first.begin(), folds[j].first.end());
                y_train.insert(y_train.end(), folds[j].second.begin(), folds[j].second.end());
            }
        }

       
        if (choice == 1)
        {
            std::cout << "Training a single decision tree on fold " << i + 1 << "/" << k << "...\n";
            int maxDepth = 60;
            int minSamplesSplit = 2;
            double minImpurityDecrease = 1e-12;

            DecisionTreeSingle single_tree(maxDepth, minSamplesSplit, minImpurityDecrease);

            auto train_start = std::chrono::high_resolution_clock::now();
            single_tree.train(X_train, y_train,0);
            auto train_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_duration = train_end - train_start;
            std::cout << "Training time: " << train_duration.count() << " seconds\n";

            auto eval_start = std::chrono::high_resolution_clock::now();
            double mse_value = 0.0;
            size_t test_size = X_test.size();
            std::vector<double> y_pred;
            y_pred.reserve(test_size);
            for (const auto &X_sample : X_test)
            {
                y_pred.push_back(single_tree.predict(X_sample));
            }
            mse_value = Math::computeLossMSE(y_test, y_pred);
            auto eval_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;

            std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";
            std::cout << "Fold " << i + 1 << "/" << k << ", Mean Squared Error (MSE): " << mse_value << "\n";

            mse_scores.push_back(mse_value);
        }
        else if (choice == 2)
        {
            std::cout << "Bagging process started on fold " << i + 1 << "/" << k << ", please wait...\n";
            int num_trees = 20;
            int max_depth = 60;
            int min_samples_split = 2;
            double min_impurity_decrease = 1e-6;
            int criteria = 0;
            int whichLossFunc = 0;

            Bagging bagging_model(num_trees, max_depth, min_samples_split, min_impurity_decrease, std::unique_ptr<LeastSquaresLoss>(), criteria, whichLossFunc);

            auto train_start = std::chrono::high_resolution_clock::now();
            bagging_model.train(X_train, y_train, criteria);
            auto train_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_duration = train_end - train_start;
            std::cout << "Training time (Bagging): " << train_duration.count() << " seconds\n";

            auto eval_start = std::chrono::high_resolution_clock::now();
            double mse_value = bagging_model.evaluate(X_test, y_test);
            auto eval_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;
            std::cout << "Evaluation time (Bagging): " << eval_duration.count() << " seconds\n";

            std::cout << "Fold " << i + 1 << "/" << k << ", Bagging Mean Squared Error (MSE): " << mse_value << "\n";

            mse_scores.push_back(mse_value);
        }
        else if (choice == 3)
        {
            std::cout << "Boosting process started on fold " << i + 1 << "/" << k << ", please wait...\n";
            int n_estimators = 20;
            int max_depth = 60;
            int min_samples_split = 2;
            double min_impurity_decrease = 1e-6;
            double learning_rate = 0.1;

            auto loss_function = std::make_unique<LeastSquaresLoss>();
            Boosting boosting_model(n_estimators, learning_rate, std::move(loss_function),
                                    max_depth, min_samples_split, min_impurity_decrease);

            auto train_start = std::chrono::high_resolution_clock::now();
            //Criteria default is 0 for MSE (faster)
            boosting_model.train(X_train, y_train, 0);
            auto train_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_duration = train_end - train_start;
            std::cout << "Training time: " << train_duration.count() << " seconds\n";

            auto eval_start = std::chrono::high_resolution_clock::now();
            double mse_value = boosting_model.evaluate(X_test, y_test);
            auto eval_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;
            std::cout << "Evaluation time: " << eval_duration.count() << " seconds\n";

            std::cout << "Fold " << i + 1 << "/" << k << ", Boosting Mean Squared Error (MSE): " << mse_value << "\n";

            mse_scores.push_back(mse_value);
        }
        else
        {
            std::cerr << "Invalid choice! Please rerun the program and choose 1, 2, or 3." << std::endl;
            return -1;
        }
    }

    
    double average_mse = std::accumulate(mse_scores.begin(), mse_scores.end(), 0.0) / mse_scores.size();
    std::cout << "Average Mean Squared Error (MSE) over " << k << " folds: " << average_mse << std::endl;

    return 0;
}
