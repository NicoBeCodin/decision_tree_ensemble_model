#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include "data_clean.h"



// Compile with g++ -o data_clean_tests data_clean_tests.cpp data_clean.cpp ../functions_tree/math_functions.cpp -std=c++17



// Helper function to create a temporary CSV file
void createTestCSV(const std::string& filePath, const std::string& header, const std::vector<std::vector<double>>& data) {
    std::ofstream file(filePath);

    // Write header
    file << header << "\n";

    // Write rows
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

// Helper function to display dataset for debugging
void printDataset(const std::vector<double>& dataset, int rowLength) {
    for (size_t i = 0; i < dataset.size(); i += rowLength) {
        for (int j = 0; j < rowLength; ++j) {
            std::cout << dataset[i + j] << " ";
        }
        std::cout << "\n";
    }
}

// Test the readCSV function
void testReadCSV() {
    std::string filePath = "test.csv";
    std::string header = "p5,p6,matrix_size_x,matrix_size_y,p2,p1,p8,p7,p3,p4,performance";
    std::vector<std::vector<double>> data = {
        {18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217},
        {24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895},
        {30, 28, 1619, 1857, 163, 102, 71, 71, 16, 25, 0.0272806}
    };

    createTestCSV(filePath, header, data);

    std::string readHeader;
    int rowLength;
    std::vector<double> dataset = readCSV(filePath, readHeader, rowLength);

    assert(readHeader == header);
    assert(rowLength == 11);
    assert(dataset.size() == 33); // 3 rows * 11 columns

    std::cout << "testReadCSV passed!\n";
}

// Test the removeOutliers function
void testRemoveOutliers() {
    std::vector<double> dataset = {
        18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217,
        24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895,
        30, 28, 1619, 1857, 163, 102, 71, 71, 16, 25, 0.0272806
    };
    int rowLength = 11;

    // Set a threshold to remove outliers
    double threshold = 2.0;

    std::vector<double> cleanedData = removeOutliers(dataset, rowLength, threshold);

    // Assert the cleaned dataset size (only valid rows remain)
    assert(cleanedData.size() == dataset.size()); // All rows are within threshold

    std::cout << "testRemoveOutliers passed!\n";
}

// Test the removeOutliersByBinning function
void testRemoveOutliersByBinning() {
    std::vector<double> dataset = {
        18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217,
        24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895,
        30, 28, 1619, 1857, 163, 102, 71, 71, 16, 25, 0.0272806
    };
    int rowLength = 11;
    int numBins = 2;
    double zThreshold = 2.0;

    std::vector<double> cleanedData = removeOutliersByBinning(dataset, rowLength, numBins, zThreshold);

    // Ensure rows remain valid after binning and outlier removal
    assert(cleanedData.size() <= dataset.size());

    std::cout << "testRemoveOutliersByBinning passed!\n";
}

// Test the writeCSV function
void testWriteCSV() {
    std::string filePath = "output.csv";
    std::string header = "p5,p6,matrix_size_x,matrix_size_y,p2,p1,p8,p7,p3,p4,performance";
    std::vector<double> dataset = {
        18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217,
        24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895
    };
    int rowLength = 11;

    writeCSV(filePath, header, dataset, rowLength);

    // Read back the written file
    std::string readHeader;
    int readRowLength;
    std::vector<double> readDataset = readCSV(filePath, readHeader, readRowLength);

    assert(readHeader == header);
    assert(readRowLength == rowLength);
    assert(readDataset == dataset);

    std::cout << "testWriteCSV passed!\n";
}

int main() {
    testReadCSV();
    testRemoveOutliers();
    testRemoveOutliersByBinning();
    testWriteCSV();

    std::cout << "All tests passed successfully!\n";
    return 0;
}
