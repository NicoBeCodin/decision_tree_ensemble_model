#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include "../functions_io/functions_io.h"

using namespace std;

// 定义树节点的数据结构
struct TreeNode {
    int feature_index;     // 当前节点分裂的特征索引
    float threshold;       // 分裂点
    float value;           // 叶节点的预测值（如果是叶节点）
    TreeNode* left;        // 左子树
    TreeNode* right;       // 右子树

    TreeNode() : feature_index(-1), threshold(0), value(0), left(nullptr), right(nullptr) {}
};

// 计算一组值的均方误差（MSE）
float calculateMSE(const vector<float>& values) {
    if (values.empty()) return 0.0;
    float mean = accumulate(values.begin(), values.end(), 0.0) / values.size();
    float mse = 0.0;
    for (float val : values) {
        mse += (val - mean) * (val - mean);
    }
    return mse / values.size();
}

// 将数据集分成两个子集，左子集包含满足分裂条件的数据，右子集包含不满足的数据
void splitDataset(const vector<vector<float>>& X, const vector<float>& y, int feature_index, float threshold,
                  vector<vector<float>>& left_X, vector<vector<float>>& right_X,
                  vector<float>& left_y, vector<float>& right_y) {
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature_index] < threshold) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }
}

// 查找当前数据集上最佳的分裂点
pair<int, float> findBestSplit(const vector<vector<float>>& X, const vector<float>& y) {
    int best_feature = -1;
    float best_threshold = 0.0;
    float best_mse = numeric_limits<float>::max();  // 初始化为最大值

    for (int feature_index = 0; feature_index < X[0].size(); ++feature_index) {
        // 获取当前特征的所有值
        vector<float> feature_values;
        for (const auto& row : X) {
            feature_values.push_back(row[feature_index]);
        }
        sort(feature_values.begin(), feature_values.end());

        // 遍历所有可能的分裂点（相邻值的中点）
        for (int i = 1; i < feature_values.size(); ++i) {
            float threshold = (feature_values[i - 1] + feature_values[i]) / 2;

            // 根据当前分裂点划分数据
            vector<vector<float>> left_X, right_X;
            vector<float> left_y, right_y;
            splitDataset(X, y, feature_index, threshold, left_X, right_X, left_y, right_y);

            // 计算加权 MSE
            float left_mse = calculateMSE(left_y);
            float right_mse = calculateMSE(right_y);
            float weighted_mse = (left_mse * left_y.size() + right_mse * right_y.size()) / y.size();

            // 更新最佳分裂点
            if (weighted_mse < best_mse) {
                best_mse = weighted_mse;
                best_feature = feature_index;
                best_threshold = threshold;
            }
        }
    }
    return {best_feature, best_threshold};
}

// 创建一个叶节点
TreeNode* createLeaf(const vector<float>& y) {
    TreeNode* leaf = new TreeNode();
    leaf->value = accumulate(y.begin(), y.end(), 0.0) / y.size();  // 叶节点的预测值为目标值的均值
    return leaf;
}

// 递归构建决策树
TreeNode* buildTree(const vector<vector<float>>& X, const vector<float>& y, int depth, int max_depth) {
    if (depth >= max_depth || y.size() <= 1) {
        return createLeaf(y);  // 如果达到最大深度或样本数过少，返回叶节点
    }

    // 查找最佳分裂点
    pair<int, float> best_split = findBestSplit(X, y);
    int best_feature = best_split.first;
    float best_threshold = best_split.second;

    if (best_feature == -1) {
        return createLeaf(y);  // 如果没有有效分裂，返回叶节点
    }

    // 根据最佳分裂点划分数据集
    vector<vector<float>> left_X, right_X;
    vector<float> left_y, right_y;
    splitDataset(X, y, best_feature, best_threshold, left_X, right_X, left_y, right_y);

    // 创建当前节点并递归构建子树
    TreeNode* node = new TreeNode();
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = buildTree(left_X, left_y, depth + 1, max_depth);
    node->right = buildTree(right_X, right_y, depth + 1, max_depth);

    return node;
}

// 使用决策树进行预测
float predict(TreeNode* node, const vector<float>& sample) {
    if (!node->left && !node->right) {
        return node->value;  // 如果是叶节点，返回预测值
    }

    if (sample[node->feature_index] < node->threshold) {
        return predict(node->left, sample);  // 递归左子树
    } else {
        return predict(node->right, sample);  // 递归右子树
    }
}

// 计算均方误差 (MSE)
float computeMSE(const vector<float>& true_values, const vector<float>& predicted_values) {
    float mse = 0.0;
    for (size_t i = 0; i < true_values.size(); ++i) {
        mse += (true_values[i] - predicted_values[i]) * (true_values[i] - predicted_values[i]);
    }
    return mse / true_values.size();
}

int main() {
    // 使用 functions_io.h 中的函数从 CSV 文件中读取训练数据
    string train_csv_file = "../datasets/sample_400_rows.csv";  // 替换为训练 CSV 文件名
    vector<vector<string>> train_content = openCSV(train_csv_file);

    // 假设第一行为表头，提取表头
    vector<string> train_header = train_content[0];
    
    // 获取特征矩阵和 performance 值
    Matrix train_parameters = processParametersCSV(train_content);  // 转换为整数矩阵
    vector<float> train_results = processResultsCSV(train_content);  // 获取 performance 列

    // 创建浮点矩阵 X
    vector<vector<float>> X(train_parameters.size(), vector<float>(train_parameters[0].size()));

    // 将 Matrix 转换为 vector<vector<float>>
    for (size_t i = 0; i < train_parameters.size(); ++i) {
        for (size_t j = 0; j < train_parameters[i].size(); ++j) {
            X[i][j] = static_cast<float>(train_parameters[i][j]);  // 显式转换 int -> float
        }
    }

    // 构建决策树，设置最大深度为 3
    int max_depth = 10;
    TreeNode* root = buildTree(X, train_results, 0, max_depth);

    // 使用 functions_io.h 从 CSV 文件中读取测试数据
    string test_csv_file = "../datasets/sample_100_rows.csv";  // 替换为测试 CSV 文件名
    vector<vector<string>> test_content = openCSV(test_csv_file);

    // 获取测试集特征矩阵和 performance 值
    Matrix test_parameters = processParametersCSV(test_content);
    vector<float> test_results = processResultsCSV(test_content);

    // 将测试集 Matrix 转换为 vector<vector<float>>
    vector<vector<float>> test_X(test_parameters.size(), vector<float>(test_parameters[0].size()));
    for (size_t i = 0; i < test_parameters.size(); ++i) {
        for (size_t j = 0; j < test_parameters[i].size(); ++j) {
            test_X[i][j] = static_cast<float>(test_parameters[i][j]);
        }
    }

    // 对测试集进行预测并存储结果
    vector<float> predicted_values;
    for (const auto& sample : test_X) {
        float prediction = predict(root, sample);
        predicted_values.push_back(prediction);
    }

    // 计算并输出 MSE
    float mse = computeMSE(test_results, predicted_values);
    cout << "测试集上的均方误差 (MSE): " << mse << endl;

    return 0;
}
