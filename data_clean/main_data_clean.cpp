#include "data_clean.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <inputFilePath> <outputFilePath>" << std::endl;
        return 1;
    }

    std::string inputFilePath = argv[1]; // Input CSV file path
    std::string outputFilePath = argv[2]; // Output CSV file path
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
