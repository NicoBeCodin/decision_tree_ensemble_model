#include "functions_tree.h"
#include <iostream>  // 引入iostream头文件以使用标准库中的输出函数
#include <vector>    // 引入vector头文件
#include <random>    // 引入random头文件
#include <algorithm> // 引入algorithm头文件以使用std::shuffle

namespace tree {

typedef std::vector<std::vector<int>> Matrix;  // 定义Matrix为二维整数向量的别名

// 计算整数数组的方差
float calculateVariance(const std::vector<int>& result_values) {
    if (result_values.empty()) {
        printf("result_values empty, can't calculate average! returning 0.0\n");
        return 0.0;
    }
    float mean = 0.0;
    for (float val : result_values) {
        mean += val;  // 计算总和
    }
    mean /= (float)result_values.size();  // 计算平均值

    printf("Mean value: %f\n", mean);  // 打印平均值

    float variance = 0.0;
    for (int val : result_values) {
        variance += ((float)val - mean) * ((float)val - mean);  // 计算方差
    }

    float result = variance / (float)result_values.size();
    printf("Variance: %f\n", result);  // 打印方差

    return result;  // 返回方差
}

// 获取某特征列的最大值
int getMaxFeature(Matrix values, int feature_index) {
    int max = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i][feature_index] > max) max = values[i][feature_index];  // 找到最大值
    }
    return max;
}

// 获取某特征列的最小值
int getMinFeature(Matrix values, int feature_index) {
    int min = 999999;  // 假设最大值为999999
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i][feature_index] < min) min = values[i][feature_index];  // 找到最小值
    }
    return min;
}

// 计算某特征列的平均值
float getMeanFeature(Matrix values, int feature_index) {
    int mean = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        mean += values[i][feature_index];  // 计算总和
    }
    float average = (float)mean / (float)values.size();  // 返回平均值
    return average;
}

// 生成唯一的随机数，用作随机抽样的行索引
std::vector<int> drawUniqueNumbers(int n, int rows) {
    if (n > rows + 1) {
        printf("Row number is smaller than sample size, setting n = rows");
        n = rows;
    }

    std::vector<int> numbers(rows);  // 创建一个行数大小的向量
    for (int i = 0; i < rows; ++i) {
        numbers[i] = i;  // 初始化行号
    }
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(numbers.begin(), numbers.end(), g);  // 随机打乱行号
    std::vector<int> result(numbers.begin(), numbers.begin() + n);  // 返回前n个作为结果
    return result;
}

// 比较不同特征的最佳阈值，并返回加权方差最小的那个
Threshold compareThresholds(std::vector<Threshold> thresholds) {
    Threshold best_threshold = thresholds[0];

    for (size_t i = 1; i < thresholds.size(); ++i) {
        if (thresholds[i].weighted_variance < best_threshold.weighted_variance) {
            best_threshold = thresholds[i];  // 更新最佳阈值
        }
    }
    return best_threshold;
}

// 为某特征找到最小化方差的最佳分裂阈值
Threshold bestThresholdColumn(Matrix values, std::vector<float> results, int column_index) {
    int best_threshold = 0;
    float min_weighted_variance = 999999.0;
    for (size_t i = 0; i < values.size(); ++i) {
        int threshold = values[i][column_index];
        std::vector<int> left;
        std::vector<int> right;

        for (size_t j = 0; j < values.size(); ++j) {
            if (values[j][column_index] < threshold) {
                left.push_back(values[j][column_index]);
            } else {
                right.push_back(values[j][column_index]);
            }
        }
        float weighted_variance = (calculateVariance(left) * (float)left.size() + calculateVariance(right) * (float)right.size()) / values.size();
        if (weighted_variance < min_weighted_variance) {
            min_weighted_variance = weighted_variance;
            best_threshold = threshold;  // 更新最佳阈值
        }
    }

    return Threshold{column_index, best_threshold, min_weighted_variance};
}

// 实现随机采样，获取最佳分裂阈值，而不是尝试所有可能的阈值
Threshold findBestSplitRandom(Matrix values, std::vector<float> results, int sample_size) {
    Matrix sample_tab;
    std::vector<float> sample_results;
    std::vector<int> tab_indexes = drawUniqueNumbers(sample_size, values.size());
    for (int index : tab_indexes) {
        sample_tab.push_back(values[index]);
        sample_results.push_back(results[index]);
    }

    std::vector<Threshold> feature_threshold;
    for (size_t i = 0; i < values[0].size(); ++i) {
        feature_threshold.push_back(bestThresholdColumn(sample_tab, sample_results, i));
    }
    Threshold best_threshold = compareThresholds(feature_threshold);
    return best_threshold;
}

// 根据最佳分裂阈值对数据进行分割
std::vector<int> splitOnThreshold(Threshold threshold, Matrix values) {
    std::vector<int> goRight(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i][threshold.feature_index] < threshold.value) {
            goRight[i] = 0;  // 左子树
        } else {
            goRight[i] = 1;  // 右子树
        }
    }
    return goRight;
}


// 创建初始节点，进行数据分裂
Node nodeInitiate(Matrix parameters, std::vector<float> results) {
    Node initialNode;

    Threshold nodeThreshold = findBestSplitRandom(parameters, results, 30);
    initialNode.threshold = nodeThreshold;
    initialNode.isLeaf = false;
    initialNode.nodeDepth = 1;

    Matrix leftValues;
    std::vector<float> leftResults;
    Matrix rightValues;
    std::vector<float> rightResults;

    std::vector<int> goRightIndex = splitOnThreshold(nodeThreshold, parameters);
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (goRightIndex[i] == 0) {
            leftValues.push_back(parameters[i]);
            leftResults.push_back(results[i]);
        } else {
            rightValues.push_back(parameters[i]);
            rightResults.push_back(results[i]);
        }
    }
    return initialNode;
}

// 递归构建子节点
Node nodeBuilder(Node parentNode) {
    Node node;
    // 递归调用构建左右子树
    return node;
}

}  // 命名空间 tree 结束
