#include "functions_io/functions_io.h"
#include "functions_tree/decision_tree_single.h"
#include "functions_tree/math_functions.h"
#include "ensemble_bagging/bagging.h" // 添加 Bagging 头文件
#include <iostream>
#include <chrono>

int main()
{
    DataIO data_io;
    auto [X, y] = data_io.readCSV("../datasets/cleaned_data.csv");
    if (X.empty() || y.empty())
    {
        std::cerr << "Error: Unable to read data. Please check the file path." << std::endl;
        return -1;
    }

    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    std::cout << "Choose the method you want to use: \n";
    std::cout << "1: Bagging\n";
    std::cout << "2: Single Decision Tree\n";
    int choice;
    std::cin >> choice;

    if (choice == 1)
    {
        std::cout << "Bagging process started, please wait...\n";
        int num_trees=10;
        int max_depth = 20;
        int min_samples_split = 4;
        double min_impurity_decrease = 1e-5;

        Bagging bagging_model(num_trees, max_depth, min_samples_split, min_impurity_decrease);




        auto train_start = std::chrono::high_resolution_clock::now();
        bagging_model.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training completed in: " << train_duration.count() << " s\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = bagging_model.evaluate(X_test, y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "Evaluation completed in: " << eval_duration.count() << " s\n";

        std::cout << "Bagging (MSE): " << mse_value << "\n";
    }
    else if (choice == 2)
    {
        std::cout << "Single Decision Tree process started, please wait...\n";
        int maxDepth = 60;
        int minSamplesSplit = 2;
        double minImpurityDecrease = 1e-12;

        DecisionTreeSingle single_tree(maxDepth, minSamplesSplit, minImpurityDecrease);

        auto train_start = std::chrono::high_resolution_clock::now();
        single_tree.train(X_train, y_train);
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end - train_start;
        std::cout << "Training completed in: " << train_duration.count() << " s\n";

        auto eval_start = std::chrono::high_resolution_clock::now();
        double mse_value = 0.0;
        size_t test_size = X_test.size();
        std::vector<double> y_pred;
        y_pred.reserve(test_size);
        for (const auto& X_sample : X_test)
        {
            y_pred.push_back(single_tree.predict(X_sample));
        }
        mse_value = Math::computeLoss(y_test, y_pred);
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;

        std::cout << "Evaluation completed in: " << eval_duration.count() << " s\n";
        std::cout << "Decision Tree Single (MSE): " << mse_value << "\n";

        std::cout << "Would you like to save this tree? (0 = no, 1 = yes)" << std::endl;
        int answer = 0;
        std::cin >> answer;
        if (answer == 1)
        {
            std::cout << "Please type the name you want to give to the .txt file: \n";
            std::string filename;
            std::cin >> filename;
            std::cout << "Saving tree as: " << filename << "\n";
            single_tree.saveTree(filename);
        }
    }
    else
    {
        std::cerr << "Invalid choice! Please select 1 or 2.\n";
        return -1;
    }

    return 0;
}
