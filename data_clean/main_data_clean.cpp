#include "data_clean.h"

int main() {
    std::string inputFilePath = "/mnt/d/Desktop/decision_tree_ensemble_model/datasets/15k_random.csv"; // Input CSV file path
    std::string outputFilePath = "/mnt/d/Desktop/decision_tree_ensemble_model/datasets/cleaned_data.csv"; // Output CSV file path
    std::string header;  // To save the header
    int numBins = 50;  // Number of bins for binning


    std::vector<DataRow> dataset = readCSV(inputFilePath, header);
    if (dataset.empty()) {
        std::cerr << "Error: Unable to read data from " << inputFilePath << std::endl;
        return -1;
    }

    // cleaning by the Z function
    std::vector<DataRow> cleanedData = removeOutliers(dataset);
    std::cout << "Z-score cleaning completed." << std::endl;

    // cleaning bag
    cleanedData = removeOutliersByBinning(cleanedData, numBins);
    std::cout << "Binning-based cleaning completed." << std::endl;

    // write the data after cleaning into the new csv
    writeCSV(outputFilePath, header, cleanedData);

    std::cout << "Cleaned data has been saved to " << outputFilePath << std::endl;

    return 0;
}
