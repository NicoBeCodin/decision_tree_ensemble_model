#include "functions_io/functions_io.h"
#include "functions_tree/regression_tree.h"
#include "ensemble_boosting/boosting.h"
#include "ensemble_boosting/loss_function.h"
#include "ensemble_bagging/bagging.h"
#include <iostream>

int main() {
    // 加载数据
    DataIO data_io;
    auto [X, y] = data_io.readCSV("/home/yifan/桌面/CHPS_M1/M1_projet/05_11/decision_tree_ensemble_model/datasets/cleaned_data.csv");
    if (X.empty() || y.empty()) {
        std::cerr << "Failed to load data." << std::endl;
        return -1;
    }

    // 划分训练集和测试集
    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    // 选择要运行的模型
    std::cout << "请选择要运行的模型: \n";
    std::cout << "1: 单一回归树\n";
    std::cout << "2: Bagging\n";
    std::cout << "3: Boosting\n";
    int choice;
    std::cin >> choice;

    if (choice == 1) {
        // 单一回归树
        int maxDepth = 5;
        MeanSquaredError mse;
        RegressionTree reg_tree(maxDepth, &mse);

        // 训练模型
        reg_tree.train(X_train, y_train);

        // 评估模型
        double mse_value = reg_tree.evaluate(X_test, y_test);
        std::cout << "单一回归树的测试集均方误差 (MSE): " << mse_value << std::endl;
    } else if (choice == 2) {
        // Bagging
        int num_trees = 10;
        int max_depth = 5;
        MeanSquaredError mse;
        Bagging bagging_model(num_trees, max_depth, &mse);

        // 训练模型
        bagging_model.train(X_train, y_train);

        // 评估模型
        double mse_value = bagging_model.evaluate(X_test, y_test);
        std::cout << "Bagging 的测试集均方误差 (MSE): " << mse_value << std::endl;
    } else if (choice == 3) {
        // Boosting
        int n_estimators = 10;
        int max_depth = 3;
        double learning_rate = 0.1;
        MeanSquaredError mse;  // 分裂标准
        auto loss_function = std::make_unique<LeastSquaresLoss>();
        Boosting boosting_model(n_estimators, max_depth, learning_rate, &mse, std::move(loss_function));

        // 训练模型
        boosting_model.train(X_train, y_train);

        // 评估模型
        double mse_value = boosting_model.evaluate(X_test, y_test);
        std::cout << "Boosting 的测试集均方误差 (MSE): " << mse_value << std::endl;
    } else {
        std::cerr << "无效的选择，请输入 1, 2 或 3." << std::endl;
        return -1;
    }

    return 0;
}
