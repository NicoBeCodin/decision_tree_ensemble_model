#include "functions_io/functions_io.h"
#include "functions_tree/decision_tree_single.h"
#include "functions_tree/math_functions.h" // 添加这一行
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

    std::cout << "Decision Tree Single process started, please wait...\n";
    int maxDepth = 60;
    int minSamplesSplit = 2;
    double minImpurityDecrease = 1e-12;

    DecisionTreeSingle single_tree(maxDepth, minSamplesSplit, minImpurityDecrease);

    auto train_start = std::chrono::high_resolution_clock::now();
    single_tree.train(X_train, y_train);
    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end - train_start;
    std::cout << "Training: " << train_duration.count() << " s" << std::endl;

    auto eval_start = std::chrono::high_resolution_clock::now();
    double mse_value = 0.0;
    size_t test_size = X_test.size();
    std::vector<double> y_pred;
    y_pred.reserve(test_size);
    for (const auto& X_sample : X_test)
    {
        y_pred.push_back(single_tree.predict(X_sample));
    }
    mse_value = Math::computeLoss(y_test, y_pred); // 计算损失
    auto eval_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eval_duration = eval_end - eval_start;

    std::cout << "Evaluation: " << eval_duration.count() << " s" << std::endl;
    std::cout << "Decision Tree Single (MSE): " << mse_value << std::endl;

    std::cout << "Would you like to save this tree? (0 = no, 1 = yes)" << std::endl;
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

    return 0;
}
