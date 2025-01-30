#ifndef DATA_CLEAN_H
#define DATA_CLEAN_H

#include <iostream>
#include <vector>
#include <string>
#include "../functions/tree/math_functions.h"

// Function declarations

std::vector<double> readCSV(const std::string& filePath, std::string& header, int& rowLength);
std::vector<double> removeOutliers(const std::vector<double>& dataset, int rowLength, double threshold = 3.0);
std::vector<int> equalFrequencyBinning(const std::vector<double>& data, int numBins);
std::vector<double> removeOutliersByBinning(const std::vector<double>& dataset, int rowLength, int numBins, double zThreshold = 3.0);
void writeCSV(const std::string& filePath, const std::string& header, const std::vector<double>& dataset, int rowLength);

#endif // DATA_CLEAN_H
