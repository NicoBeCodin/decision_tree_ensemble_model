#include "data_clean.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>


// The DataRow structure is no longer defined here, it is directly used from the definition in data_clean.h



// Function to calculate the standard deviation


// Function to read a CSV file and parse it into a vector of DataRow structures
std::vector<DataRow> readCSV(const std::string& filePath, std::string& header) {
    std::vector<DataRow> dataset;
    std::ifstream file(filePath);
    std::string line;

    // Read and save the header
    std::getline(file, header);

    // Read each row of data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        DataRow row;
        std::string value;

        // Read each column's value
        while (std::getline(ss, value, ',')) {
            row.values.push_back(value);  // Save data of all columns
        }

        // Assuming matrix_size_x, matrix_size_y, and performance are the 2nd, 3rd, and last columns, respectively
        if (row.values.size() > 2) {
            row.matrix_size_x = std::stod(row.values[1]);
            row.matrix_size_y = std::stod(row.values[2]);
        }
        row.performance = std::stod(row.values.back());  // Assume performance is the last column

        dataset.push_back(row);
    }
    return dataset;
}

// Function to remove outliers based on Z-scores
std::vector<DataRow> removeOutliers(const std::vector<DataRow>& dataset, double threshold) {
    std::vector<double> performances;
    for (const auto& row : dataset) {
        performances.push_back(row.performance);
    }
    

    double mean = Math::calculateMean(performances);
    double stdDev = Math::calculateStdDev(performances, mean);

    // Remove rows where the Z-score exceeds the threshold
    std::vector<DataRow> cleanedData;
    for (const auto& row : dataset) {
        double z = (row.performance - mean) / stdDev;
        if (std::abs(z) <= threshold) {
            cleanedData.push_back(row);  // Keep the entire row, not just the performance column
        }
    }

    return cleanedData;
}

// Equal-frequency binning: Divides data into numBins bins with an equal number of data points
std::vector<int> equalFrequencyBinning(std::vector<double> data, int numBins) {
    std::vector<int> binnedData;
    std::sort(data.begin(), data.end());  // Sort the data
    int binSize = data.size() / numBins;  // Number of data points per bin

    for (size_t i = 0; i < data.size(); ++i) {
        int binIndex = std::min(static_cast<int>(i / binSize), numBins - 1);
        binnedData.push_back(binIndex);  // Record the bin each data point belongs to
    }

    return binnedData;
}

// Bin the data based on matrix size X and Y, and perform Z-score outlier detection within each bin
std::vector<DataRow> removeOutliersByBinning(const std::vector<DataRow>& dataset, int numBins, double zThreshold) {
    // Extract matrix size X and Y values for binning
    std::vector<double> matrixSizesX, matrixSizesY;
    for (const auto& row : dataset) {
        matrixSizesX.push_back(row.matrix_size_x);
        matrixSizesY.push_back(row.matrix_size_y);
    }

    // Perform equal-frequency binning on X and Y
    std::vector<int> xBins = equalFrequencyBinning(matrixSizesX, numBins);
    std::vector<int> yBins = equalFrequencyBinning(matrixSizesY, numBins);

    // Perform Z-score detection on performance data within each bin
    std::vector<DataRow> cleanedData;
    for (int bin = 0; bin < numBins; ++bin) {
        std::vector<double> performancesInBin;
        std::vector<DataRow> rowsInBin;

        // Extract data points within the same bin
        for (size_t i = 0; i < dataset.size(); ++i) {
            if (xBins[i] == bin || yBins[i] == bin) {  // Ensure the same bin for X or Y
                performancesInBin.push_back(dataset[i].performance);
                rowsInBin.push_back(dataset[i]);
            }
        }

        if (performancesInBin.empty()) continue;  // Skip empty bins

        // Calculate Z-scores within the bin
        double mean = Math::calculateMean(performancesInBin);
        double stdDev = Math::calculateStdDev(performancesInBin, mean);
        for (size_t i = 0; i < rowsInBin.size(); ++i) {
            double z = (performancesInBin[i] - mean) / stdDev;
            if (std::abs(z) <= zThreshold) {
                cleanedData.push_back(rowsInBin[i]);  // Only keep rows with Z-scores within the threshold
            }
        }
    }

    return cleanedData;
}

// Function to write the cleaned data into a new CSV file, keeping the header
void writeCSV(const std::string& filePath, const std::string& header, const std::vector<DataRow>& dataset) {
    std::ofstream outFile(filePath);

    // Write the header
    outFile << header << "\n";

    // Write the data
    for (const auto& row : dataset) {
        for (size_t i = 0; i < row.values.size(); ++i) {
            outFile << row.values[i];
            if (i < row.values.size() - 1) {
                outFile << ","; // Column separator
            }
        }
        outFile << "\n"; // End of row
    }

    outFile.close();
}
