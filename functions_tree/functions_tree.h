#ifndef FUNCTIONS_TREE_H
#define FUNCTIONS_TREE_H

#include <vector>
#include <random>
#include <iostream>

#include "../functions_io/functions_io.h"

// Define Matrix as a type alias for a two-dimensional vector of integers
typedef std::vector<std::vector<int>> Matrix;

// Define the Threshold structure to store information about the split threshold
struct Threshold {
    int feature_index;
    int value;
    float weighted_variance;

    Threshold(): feature_index(-1), value(-999), weighted_variance(999.99) {}
    Threshold(int f, int v, float wv): feature_index(f), value(v), weighted_variance(wv) {}
};

// Define the Node structure, representing a node in the tree
struct Node {
    bool isLeaf;
    float value; // If leaf node, store predicted value (mean of target values)
    Threshold threshold; // Threshold to split on
    int nodeDepth; 
    Node* left; // Left child node
    Node* right; // Right child node
    // Information
    std::vector<int> adress; // 0 is left, 1 is right; list gives the address of nodes: adress.size() == depth
    int data_size; // How many rows go through the node

    Node(): isLeaf(false), value(0.0), threshold(Threshold()), nodeDepth(0), left(nullptr), right(nullptr), adress({}), data_size(0) {}
};

// Calculate the variance of an array of integers 
float calculateVariance(const std::vector<int>& result_values);

int getMaxFeature(Matrix& values, int feature_index);

int getMinFeature(Matrix& values, int feature_index);

float getMeanFeature(Matrix& values, int feature_index);

std::vector<int> drawUniqueNumbers(int n, int rows);

Threshold compareThresholds(std::vector<Threshold>& thresholds);

Threshold bestThresholdColumn(Matrix& values, std::vector<float>& results, int column_index);

Threshold findBestSplitRandom(Matrix& values, std::vector<float>& results, int sample_size);

std::vector<int> splitOnThreshold(Threshold& threshold, Matrix& values);

Node* nodeInitiate(Matrix& parameters, std::vector<float>& results);

Node* nodeBuilder(Node* parentNode, Matrix& parameters, std::vector<float>& results, bool right);

// namespace tree

#endif // FUNCTIONS_TREE_H