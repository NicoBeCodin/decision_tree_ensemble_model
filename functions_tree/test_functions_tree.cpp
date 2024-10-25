#include "functions_tree.h"
#include <iostream>
#include <cassert>  // For assertion testing

using namespace std;

void testCalculateVariance() {
    vector<int> values1 = {1, 2, 3, 4, 5};
    float variance1 = tree::calculateVariance(values1);
    assert(variance1 == 2.0);  // Expected variance is 2.0

    vector<int> values2 = {5, 5, 5, 5};
    float variance2 = tree::calculateVariance(values2);
    assert(variance2 == 0.0);  // All values are the same, variance should be 0

    vector<int> empty_values = {};
    float variance_empty = tree::calculateVariance(empty_values);
    assert(variance_empty == 0.0);  // Variance of an empty array should be 0

    cout << "testCalculateVariance passed!" << endl;
}

void testGetMaxMinFeature() {
    tree::Matrix matrix = {{1, 2}, {3, 4}, {5, 6}};
    
    int maxFeature = tree::getMaxFeature(matrix, 1);
    int minFeature = tree::getMinFeature(matrix, 0);

    assert(maxFeature == 6);  // The maximum value in the second column should be 6
    assert(minFeature == 1);  // The minimum value in the first column should be 1

    cout << "testGetMaxMinFeature passed!" << endl;
}

void testNodeInitiate() {
    tree::Matrix matrix = {{1, 2}, {3, 4}, {5, 6}};
    vector<float> results = {0.1, 0.2, 0.3};

    tree::Node node = tree::nodeInitiate(matrix, results);

    // Check if the node is created correctly
    assert(!node.isLeaf);  // The initialized node should not be a leaf node
    assert(node.nodeDepth == 1);  // The initial node depth should be 1

    cout << "testNodeInitiate passed!" << endl;
}

void testSplitOnThreshold() {
    tree::Matrix matrix = {{1, 2}, {3, 4}, {5, 6}};
    tree::Threshold threshold = {1, 4, 0.5};

    vector<int> goRight = tree::splitOnThreshold(threshold, matrix);

    // Check if the split is correct
    assert(goRight[0] == 0);  // The first element should go to the left subtree
    assert(goRight[1] == 0);  // The second element should go to the left subtree
    assert(goRight[2] == 1);  // The third element should go to the right subtree

    cout << "testSplitOnThreshold passed!" << endl;
}

int main() {
    // Call the test functions
    testCalculateVariance();
    testGetMaxMinFeature();
    testNodeInitiate();
    testSplitOnThreshold();

    cout << "All tests passed!" << endl;
    return 0;
}
