#ifndef FUNCTIONS_TREE_H
#define FUNCTIONS_TREE_H

#include <vector>
#include <random>
#include <iostream>

#include "functions_io.h"



// 定义Matrix为二维整数向量的别名
typedef std::vector<std::vector<int>> Matrix;

// 定义Threshold结构体，用于存储分裂阈值的信息
struct Threshold {
    int feature_index;
    int value;
    float weighted_variance;

    Threshold(): feature_index(-1), value(-999), weighted_variance(999.99) {}
    Threshold(int f, int v, float wv): feature_index(f), value(v), weighted_variance(wv) {}
};

// 定义Node结构体，表示树的节点
struct Node {
    bool isLeaf;
    float value; //If leaf node, store predicted value (mean of target values)
    Threshold threshold; //Threshold to split on
    int nodeDepth; 
    Node* left; //Left child node
    Node* right; //Right child node
    //Information
    vector<int> adress; //0 is left 1 is right, list gives the adress of nodes: adress.size() == depth
    int data_size; //How many rows go through node

    Node(): isLeaf(false), value(0.0), threshold(Threshold()), nodeDepth(0), left(nullptr), right(nullptr), adress({}), data_size(0) {}
};

//calculate variance of array of ints 
float calculateVariance(const vector<int>& result_values);

int getMaxFeature(Matrix& values, int feature_index);

int getMinFeature(Matrix& values, int feature_index);

float getMeanFeature(Matrix& values, int feature_index);

vector<int> drawUniqueNumbers(int n, int rows);

Threshold compareThresholds(vector<Threshold>& thresholds);

Threshold bestThresholdColumn(Matrix& values, vector<float>& results, int column_index);

Threshold findBestSplitRandom(Matrix& values, vector<float>& results, int sample_size);

vector<int> splitOnThreshold(Threshold& threshold, Matrix& values);

Node* nodeInitiate(Matrix& parameters, vector<float>& results);

Node* nodeBuilder(Node* parentNode, Matrix& parameters, vector<float>& results, bool right);

// namespace tree

#endif // FUNCTIONS_TREE_H
