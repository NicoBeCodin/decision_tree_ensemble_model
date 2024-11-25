#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H

#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <memory>
#include <tuple>

class DecisionTreeSingle
{
private:
    struct Tree
    {
        int FeatureIndex = -1;       // 分裂特征索引
        double MaxValue = 0.0;       // 分裂阈值
        double Prediction = 0.0;     // 叶节点预测值
        bool IsLeaf = false;         // 是否为叶节点
        std::unique_ptr<Tree> Left = nullptr;   // 左子树
        std::unique_ptr<Tree> Right = nullptr;  // 右子树
    };
    std::unique_ptr<Tree> Root;      // 根节点
    int MaxDepth;                    // 最大深度
    int MinLeafLarge;                // 最小叶子大小
    double MinError;                 // 最小误差

    // 已删除 SplittingCriteria* Criteria;

    // 私有成员函数
    void splitNode(Tree* Node, const std::vector<std::vector<double>>& Data,
                   const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth);

    std::tuple<int, double, double> findBestSplit(const std::vector<std::vector<double>>& Data,
                                                  const std::vector<double>& Labels, const std::vector<int>& Indices, double CurrentMSE);

    double calculateMean(const std::vector<double>& Labels, const std::vector<int>& Indices);

    double calculateMSE(const std::vector<double>& Labels, const std::vector<int>& Indices);

    std::vector<std::vector<int>> preSortFeatures(const std::vector<std::vector<double>>& Data, const std::vector<int>& Indices);

    void serializeNode(const Tree* node, std::ostream& out);

    std::unique_ptr<Tree> deserializeNode(std::istream& in);

public:
    // 构造函数：初始化最大深度、最小叶子大小和最小误差
    DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError);

    // 训练函数
    void train(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels);

    // 预测函数
    double predict(const std::vector<double>& Sample) const;

    // 保存和加载模型
    void saveTree(const std::string& filename);
    void loadTree(const std::string& filename);
};

#endif // DECISION_TREE_SINGLE_H
