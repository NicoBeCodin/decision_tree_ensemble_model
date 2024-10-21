#include "functions_tree.h"
#include <iostream>
#include <cassert>  // 用于断言测试

using namespace std;

void testCalculateVariance() {
    vector<int> values1 = {1, 2, 3, 4, 5};
    float variance1 = tree::calculateVariance(values1);
    assert(variance1 == 2.0);  // 预期的方差是2.0

    vector<int> values2 = {5, 5, 5, 5};
    float variance2 = tree::calculateVariance(values2);
    assert(variance2 == 0.0);  // 所有值相同，方差应为0

    vector<int> empty_values = {};
    float variance_empty = tree::calculateVariance(empty_values);
    assert(variance_empty == 0.0);  // 空数组方差应为0

    cout << "testCalculateVariance passed!" << endl;
}



void testGetMaxMinFeature() {
    tree::Matrix matrix = {{1, 2}, {3, 4}, {5, 6}};
    
    int maxFeature = tree::getMaxFeature(matrix, 1);
    int minFeature = tree::getMinFeature(matrix, 0);

    assert(maxFeature == 6);  // 第二列的最大值应为6
    assert(minFeature == 1);  // 第一列的最小值应为1

    cout << "testGetMaxMinFeature passed!" << endl;
}

void testNodeInitiate() {
    tree::Matrix matrix = {{1, 2}, {3, 4}, {5, 6}};
    vector<float> results = {0.1, 0.2, 0.3};

    tree::Node node = tree::nodeInitiate(matrix, results);

    // 检查节点是否创建正确
    assert(!node.isLeaf);  // 初始化节点不应是叶节点
    assert(node.nodeDepth == 1);  // 初始节点深度应为1

    cout << "testNodeInitiate passed!" << endl;
}

void testSplitOnThreshold() {
    tree::Matrix matrix = {{1, 2}, {3, 4}, {5, 6}};
    tree::Threshold threshold = {1, 4, 0.5};

    vector<int> goRight = tree::splitOnThreshold(threshold, matrix);

    // 检查是否正确分割
    assert(goRight[0] == 0);  // 第一个元素应该在左子树
    assert(goRight[1] == 0);  // 第二个元素应该在左子树
    assert(goRight[2] == 1);  // 第三个元素应该在右子树

    cout << "testSplitOnThreshold passed!" << endl;
}

int main() {
    // 调用测试函数
    testCalculateVariance();
    testGetMaxMinFeature();
    testNodeInitiate();
    testSplitOnThreshold();

    cout << "All tests passed!" << endl;
    return 0;
}
