#ifndef DATA_CLEAN_H
#define DATA_CLEAN_H

#include <iostream>
#include <vector>
#include <string>
#include "../functions_tree/math_functions.h"

// Structure representing the data of each row
struct DataRow {
    std::vector<std::string> values;
    double performance;
    double matrix_size_x;
    double matrix_size_y;
};

// Function declarations

std::vector<DataRow> readCSV(const std::string& filePath, std::string& header);
std::vector<DataRow> removeOutliers(const std::vector<DataRow>& dataset, double threshold = 3.0);
std::vector<int> equalFrequencyBinning(std::vector<double> data, int numBins);
std::vector<DataRow> removeOutliersByBinning(const std::vector<DataRow>& dataset, int numBins, double zThreshold = 3.0);
void writeCSV(const std::string& filePath, const std::string& header, const std::vector<DataRow>& dataset);

#endif // DATA_CLEAN_H
