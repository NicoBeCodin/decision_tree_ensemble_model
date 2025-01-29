#include "data_clean/data_clean.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <inputFilePath> <outputFilePath>" << std::endl;
        return 1;
    }

    std::string inputFilePath = argv[1]; // Input CSV file path
    std::string outputFilePath = argv[2]; // Output CSV file path
    std::string header;  // To save the header
    int numBins = 50;  // Number of bins for binning
    int rowLength = 11;
    int zThreshold = 3;  // Z-score threshold for outlier detection


    std::vector<double> dataset = readCSV(inputFilePath, header, rowLength);
    if (dataset.empty()) {
        std::cerr << "Error: Unable to read data from " << inputFilePath << std::endl;
        return -1;
    }

    std::cout << "Dataset loaded successfully. Number of rows: " 
              << dataset.size() / rowLength << ", Number of columns: " 
              << rowLength << std::endl;

    // cleaning by the Z function
    std::vector<double> cleanedData = removeOutliers(dataset, rowLength, zThreshold);
    std::cout << "Z-score cleaning completed." << std::endl;

    // cleaning bag
    cleanedData = removeOutliersByBinning(cleanedData,  rowLength, numBins);
    std::cout << "Binning-based cleaning completed. Remaining rows: " 
              << cleanedData.size() / rowLength << std::endl;

    // write the data after cleaning into the new csv
    writeCSV(outputFilePath, header, cleanedData, rowLength);

    std::cout << "Cleaned data has been saved to " << outputFilePath << std::endl;

    return 0;
}
