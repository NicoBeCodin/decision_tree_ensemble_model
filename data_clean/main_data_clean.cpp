#include "data_clean.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

int main() {
    std::string inputFilePath = "/Users/doriandrivet/test_Tree/decision_tree_ensemble_model/datasets/sample_400_row.csv"; // Input CSV file path
    std::string outputFilePath = "/Users/doriandrivet/test_Tree/decision_tree_ensemble_model/datasets/cleaned_data.csv"; // Output CSV file path
    std::string header;  // To save the header
    int numBins = 50;  // Number of bins for binning


    std::vector<DataRow> dataset = readCSV(inputFilePath, header);

    // cleaning by the Z function
    std::vector<DataRow> cleanedData = removeOutliers(dataset);
    std::cout << "Z 分数清理已完成。" << std::endl;

    // cleaning bag
    cleanedData = removeOutliersByBinning(cleanedData, numBins);
    std::cout << "分箱清理已完成。" << std::endl;

    // write the data after cleaning into the new csv
    writeCSV(outputFilePath, header, cleanedData);

    std::cout << "清理后的数据已保存至 " << outputFilePath << std::endl;

    return 0;
}
