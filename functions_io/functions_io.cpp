#include "functions_io.h"
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * Reads data from a CSV file.
 * @param filename The path to the CSV file
 * @return A pair containing the feature matrix and target vector; the first element is the feature matrix, and the second element is the target vector
 */
std::pair<std::vector<std::vector<double>>, std::vector<double>> DataIO::readCSV(const std::string& filename) {
    std::vector<std::vector<double>> features;  // Stores the feature matrix
    std::vector<double> labels;  // Stores the target vector

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return {features, labels};  // Return empty data
    }

    std::string line;
    bool headerSkipped = false;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;  // Stores the feature values of the current line

        // Skip the first line (assumed to be the header)
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        // Parse each column value as a double
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        // Treat the last column as the target value, and the remaining columns as features
        if (!row.empty()) {
            labels.push_back(row.back());  // Last column is the target value
            row.pop_back();  // Remove the target value column
            features.push_back(row);  // Remaining columns are features
        }
    }

    file.close();
    return {features, labels};
}

/**
 * Writes prediction results to a file.
 * @param results The vector of prediction results to write
 * @param filename The path to the file to save the results
 */
void DataIO::writeResults(const std::vector<double>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    // Write each prediction result on a new line
    for (const auto& result : results) {
        file << result << "\n";
    }

    file.close();
}
