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
    std::string inputFilePath = "/home/yifan/桌面/31_10_ppn/decision_tree_ensemble_model/datasets/15k_random.csv"; // Input CSV file path
    std::string outputFilePath = "/home/yifan/桌面/31_10_ppn/decision_tree_ensemble_model/datasets/cleaned_data.csv"; // Output CSV file path
    std::string header;  // To save the header
    int numBins = 50;  // Number of bins for binning

    // 读取数据并保存表头
    std::vector<DataRow> dataset = readCSV(inputFilePath, header);

    // 先进行 Z 分数清理
    std::vector<DataRow> cleanedData = removeOutliers(dataset);
    std::cout << "Z 分数清理已完成。" << std::endl;

    // 再进行分箱清理
    cleanedData = removeOutliersByBinning(cleanedData, numBins);
    std::cout << "分箱清理已完成。" << std::endl;

    // 将清理后的数据写入新的 CSV 文件
    writeCSV(outputFilePath, header, cleanedData);

    std::cout << "清理后的数据已保存至 " << outputFilePath << std::endl;

    return 0;
}
