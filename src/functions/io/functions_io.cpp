#include "functions_io.h"
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * Reads data from a CSV file.
 * @param filename The path to the CSV file
 * @param rowLength Reference variable to store the number of columns (features + target)
 * @return A pair containing the flattened feature matrix and target vector
 */
std::pair<std::vector<double>, std::vector<double>> DataIO::readCSV(const std::string& filename, int& rowLength) {
    std::vector<double> flattenedFeatures;  // Flattened feature matrix
    std::vector<double> labels;  // Target vector

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return {flattenedFeatures, labels};  // Return empty data
    }

    std::string line;
    bool headerSkipped = false;

    // Read the file line by line
    rowLength = 0; // Initialize row length
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Skip the first line (assumed to be the header)
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        std::vector<double> row;  // Temporary vector to store row values

        // Parse each column value as a double
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        // Treat the last column as the target value, and the remaining columns as features
        if (!row.empty()) {
            labels.push_back(row.back());  // Last column is the target value
            row.pop_back();  // Remove the target value column

            // Append the remaining feature values to the flattened feature matrix
            flattenedFeatures.insert(flattenedFeatures.end(), row.begin(), row.end());

            // Set row length only once (features + 1 for the target)
            if (rowLength == 0) {
                rowLength = row.size() + 1;
            }
        }
    }

    file.close();
    return {flattenedFeatures, labels};
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

