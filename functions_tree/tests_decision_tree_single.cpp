#include "decision_tree_single.h"
#include <iostream>
#include <vector>
#include <cassert>

// Helper function to create a flattened dataset
std::vector<double> createFlattenedDataset(const std::vector<std::vector<double>>& data) {
    std::vector<double> flattened;
    for (const auto& row : data) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

void testDecisionTree() {
    // Define dataset and labels
    std::vector<std::vector<double>> data = {
        {2.3, 1.5, 3.1},
        {1.8, 1.0, 2.8},
        {3.2, 2.5, 4.1},
        {2.0, 1.7, 3.0},
        {3.0, 2.0, 4.0}
    };
    std::vector<double> labels = {1.0, 0.8, 1.5, 1.2, 1.6};

    // Flatten the dataset
    std::vector<double> flattenedData = createFlattenedDataset(data);
    int rowLength = data[0].size();

    // Create decision tree
    DecisionTreeSingle tree(3, 2, 0.01);

    // Train the tree
    tree.train(flattenedData, rowLength, labels, 0); // Using MSE

    // Predict a sample
    std::vector<double> sample = {2.5, 1.8, 3.5};
    double prediction = tree.predict(sample);

    // Output prediction
    std::cout << "Prediction: " << prediction << std::endl;

    // Test saving and loading the tree
    tree.saveTree("tree_model.txt");
    DecisionTreeSingle loadedTree(3, 2, 0.01);
    loadedTree.loadTree("tree_model.txt");

    // Verify the loaded tree gives the same prediction
    double loadedPrediction = loadedTree.predict(sample);
    assert(prediction == loadedPrediction);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    testDecisionTree();
    return 0;
}
