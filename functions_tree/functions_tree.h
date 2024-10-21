#ifndef FUNCTIONS_TREE_H
#define FUNCTIONS_TREE_H

#include <vector>
#include <random>
#include <iostream>

namespace tree {

// 定义Matrix为二维整数向量的别名
typedef std::vector<std::vector<int>> Matrix;

// 定义Threshold结构体，用于存储分裂阈值的信息
struct Threshold {
    int feature_index;       // 特征列索引
    int value;               // 分裂阈值
    float weighted_variance; // 加权方差
};

// 定义Node结构体，表示树的节点
struct Node {
    Threshold threshold;  // 节点的分裂阈值
    bool isLeaf;          // 是否是叶子节点
    int nodeDepth;        // 节点的深度
};

// 函数声明
float calculateVariance(const std::vector<int>& result_values);  // 计算方差
int getMaxFeature(Matrix values, int feature_index);  // 获取特征列的最大值
int getMinFeature(Matrix values, int feature_index);  // 获取特征列的最小值
float getMeanFeature(Matrix values, int feature_index);  // 计算特征列的平均值
std::vector<int> drawUniqueNumbers(int n, int rows);  // 生成唯一的随机数，用于随机抽样
Threshold compareThresholds(std::vector<Threshold> thresholds);  // 比较不同阈值，找到最佳阈值
Threshold bestThresholdColumn(Matrix values, std::vector<float> results, int column_index);  // 找到某列的最佳分裂阈值
Threshold findBestSplitRandom(Matrix values, std::vector<float> results, int sample_size);  // 实现随机采样，找到最佳分裂阈值
std::vector<int> splitOnThreshold(Threshold threshold, Matrix values);  // 根据阈值分割数据
Node nodeInitiate(Matrix parameters, std::vector<float> results);  // 创建初始节点
Node nodeBuilder(Node parentNode);  // 递归构建子节点

} // namespace tree

#endif // FUNCTIONS_TREE_H
