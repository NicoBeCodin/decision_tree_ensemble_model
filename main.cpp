#include "functions_io/functions_io.h"
#include "functions_tree/regression_tree.h"
#include "functions_tree/decision_tree_single.h"
#include "ensemble_boosting/boosting.h"
#include "ensemble_boosting/loss_function.h"
#include "ensemble_bagging/bagging.h"
#include <iostream>
#include <chrono>

int main() {
    
    DataIO data_io;
    auto [X, y] = data_io.readCSV("../datasets/cleaned_data.csv");
    if (X.empty() || y.empty()) {
        std::cerr << "Suprise mother fucker!Check the adress again!." << std::endl;
        return -1;
    }


    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    std::cout << "Choise the methode u want use: \n";
    std::cout << "1: decision_tree_regression\n";
    std::cout << "2: Bagging\n";
    std::cout << "3: Boosting\n";
    std::cout << "4: decision_tree_simple\n";
    int choice;
    std::cin >> choice;

    if (choice == 1) {
    
        int maxDepth = 5;
        MeanSquaredError mse;
        RegressionTree reg_tree(maxDepth, &mse);

      
        auto train_start = std::chrono::high_resolution_clock::now();
        reg_tree.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Train: " << train_duration.count() << " s" << std::endl;

      
        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = reg_tree.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evalue: " << eval_duration.count() << " s" << std::endl;

        std::cout << " (MSE): " << mse_value << std::endl;

    } else if (choice == 2) {
        // Bagging
        int num_trees = 10;
        int max_depth = 5;
        MeanSquaredError mse;
        Bagging bagging_model(num_trees, max_depth, &mse);

        auto train_start = std::chrono::high_resolution_clock::now();
        bagging_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "train（Bagging）: " << train_duration.count() << " s" << std::endl;

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = bagging_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evalue（Bagging）: " << eval_duration.count() << " s" << std::endl;

        std::cout << "Bagging  (MSE): " << mse_value << std::endl;

    } else if (choice == 3) {
        // Boosting
        int n_estimators = 10;
        int max_depth = 5;
        double learning_rate = 0.1;
        MeanSquaredError mse;
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        Boosting boosting_model(n_estimators, max_depth, learning_rate, &mse, std::move(loss_function));

        auto train_start = std::chrono::high_resolution_clock::now();
        boosting_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "train（Boosting）: " << train_duration.count() << " s" << std::endl;

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = boosting_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evalue（Boosting）: " << eval_duration.count() << " s" << std::endl;

        std::cout << "Boosting  (MSE): " << mse_value << std::endl;

    } else if (choice == 4) {
  
    int maxDepth = 60; 
    int minSamplesSplit = 2;  
    double minImpurityDecrease = 1e-12;  
    MeanSquaredError mse;
    DecisionTreeSingle single_tree(maxDepth, minSamplesSplit, minImpurityDecrease, &mse);

    
    auto train_start = std::chrono::high_resolution_clock::now();
    single_tree.Train(X_train, y_train);
    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end - train_start;
    std::cout << "train: " << train_duration.count() << " s" << std::endl;

    
    auto eval_start = std::chrono::high_resolution_clock::now();
    double mse_value = 0.0;
    size_t test_size = X_test.size();
    for (size_t i = 0; i < test_size; ++i) {
        double prediction = single_tree.Predict(X_test[i]);
        mse_value += std::pow(prediction - y_test[i], 2);
    }
    mse_value /= test_size;
    auto eval_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eval_duration = eval_end - eval_start;

    std::cout << "decision_tree_single evalue: " << eval_duration.count() << " s" << std::endl;
    std::cout << "(MSE): " << mse_value << std::endl;

} else {
    std::cerr << "Are u an asshole?From 1 to 4!" << std::endl;
    return -1;
}


    return 0;
}
