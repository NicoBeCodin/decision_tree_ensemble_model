#include "data_clean.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

// Function to read a CSV file and parse it into a flat vector of doubles
std::vector<double> readCSV(const std::string& filePath, std::string& header, int& rowLength) {
    std::vector<double> dataset;
    std::ifstream file(filePath);
    std::string line;

    // Read and save the header
    std::getline(file, header);

    // Read each row of data
    rowLength = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Read each column's value
        while (std::getline(ss, value, ',')) {
            dataset.push_back(std::stod(value));  // Convert value to double and store it
        }

        if (rowLength == 0) {
            rowLength = dataset.size(); // Set the number of columns in the first row
        }
    }
    return dataset;
}

// Function to remove outliers based on Z-scores
std::vector<double> removeOutliers(const std::vector<double>& dataset, int rowLength, double threshold) {
    int numRows = dataset.size() / rowLength;
    std::vector<double> performances;

    // Extract performance column
    for (int i = 0; i < numRows; ++i) {
        performances.push_back(dataset[i * rowLength + rowLength - 1]);
    }

    double mean = Math::calculateMean(performances);
    double stdDev = Math::calculateStdDev(performances, mean);

    // Remove rows where the Z-score exceeds the threshold
    std::vector<double> cleanedData;
    for (int i = 0; i < numRows; ++i) {
        double z = (performances[i] - mean) / stdDev;
        if (std::abs(z) <= threshold) {
            for (int j = 0; j < rowLength; ++j) {
                cleanedData.push_back(dataset[i * rowLength + j]);
            }
        }
    }

    return cleanedData;
}

// Equal-frequency binning
std::vector<int> equalFrequencyBinning(const std::vector<double>& data, int numBins) {
    std::vector<int> binnedData;
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());

    int binSize = data.size() / numBins;
    for (size_t i = 0; i < data.size(); ++i) {
        int binIndex = std::min(static_cast<int>(i / binSize), numBins - 1);
        binnedData.push_back(binIndex);
    }

    return binnedData;
}

// Remove outliers within each bin
std::vector<double> removeOutliersByBinning(const std::vector<double>& dataset, int rowLength, int numBins, double zThreshold) {
    int numRows = dataset.size() / rowLength;
    std::vector<double> matrixSizesX, matrixSizesY, performances;

    // Extract relevant columns
    for (int i = 0; i < numRows; ++i) {
        matrixSizesX.push_back(dataset[i * rowLength + 2]); // Assuming matrix_size_x is at index 2
        matrixSizesY.push_back(dataset[i * rowLength + 3]); // Assuming matrix_size_y is at index 3
        performances.push_back(dataset[i * rowLength + rowLength - 1]); // Performance at last index
    }

    // Perform equal-frequency binning
    std::vector<int> xBins = equalFrequencyBinning(matrixSizesX, numBins);
    std::vector<int> yBins = equalFrequencyBinning(matrixSizesY, numBins);

    std::vector<double> cleanedData;
    for (int bin = 0; bin < numBins; ++bin) {
        std::vector<double> performancesInBin;
        std::vector<int> indicesInBin;

        // Extract data points within the same bin
        for (int i = 0; i < numRows; ++i) {
            if (xBins[i] == bin || yBins[i] == bin) {
                performancesInBin.push_back(performances[i]);
                indicesInBin.push_back(i);
            }
        }

        if (performancesInBin.empty()) continue;

        double mean = Math::calculateMean(performancesInBin);
        double stdDev = Math::calculateStdDev(performancesInBin, mean);

        // Keep rows within the Z-score threshold
        for (int i : indicesInBin) {
            double z = (performances[i] - mean) / stdDev;
            if (std::abs(z) <= zThreshold) {
                for (int j = 0; j < rowLength; ++j) {
                    cleanedData.push_back(dataset[i * rowLength + j]);
                }
            }
        }
    }

    return cleanedData;
}

// Write the cleaned data into a new CSV file
void writeCSV(const std::string& filePath, const std::string& header, const std::vector<double>& dataset, int rowLength) {
    std::ofstream outFile(filePath);

    // Write the header
    outFile << header << "\n";

    // Write the data
    int numRows = dataset.size() / rowLength;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < rowLength; ++j) {
            outFile << dataset[i * rowLength + j];
            if (j < rowLength - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
}