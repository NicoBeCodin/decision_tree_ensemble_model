#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>  // for accumulate

// Structure representing the data of each row
struct DataRow {
    std::vector<std::string> values;  // Save data of all columns
    double performance;
};

// Function to calculate the mean
double calculateMean(const std::vector<double>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

// Function to calculate the standard deviation
double calculateStdDev(const std::vector<double>& data, double mean) {
    double sum = 0.0;
    for (const auto& value : data) {
        sum += std::pow(value - mean, 2);
    }
    return std::sqrt(sum / data.size());
}

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

        // The last column is performance, convert it to double for calculations
        row.performance = std::stod(row.values.back());  // Assume performance is the last column
        dataset.push_back(row);
    }
    return dataset;
}

// Function to remove outliers based on Z-scores
std::vector<DataRow> removeOutliers(const std::vector<DataRow>& dataset, double threshold = 3.0) {
    std::vector<double> performances;
    for (const auto& row : dataset) {
        performances.push_back(row.performance);
    }

    double mean = calculateMean(performances);
    double stdDev = calculateStdDev(performances, mean);

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

int main() {
    std::string inputFilePath = "datasets/15k_random.csv"; // Input CSV file path
    std::string outputFilePath = "/datasets/cleaned_data.csv"; // Output CSV file path
    std::string header;  // To save the header

    // Read the data and save the header
    std::vector<DataRow> dataset = readCSV(inputFilePath, header);

    // Remove outliers with Z-scores greater than 3
    std::vector<DataRow> cleanedData = removeOutliers(dataset);

    // Write the data after cleaning into the new CSV
    writeCSV(outputFilePath, header, cleanedData);

    std::cout << "The cleaned data has been saved to " << outputFilePath << std::endl;

    return 0;
}
