#ifndef FUNCTIONS_IO_H
#define FUNCTIONS_IO_H

#include <vector>
#include <string>
#include <utility>

class DataIO {
public:
    /**
     * Reads data from a CSV file.
     * @param filename The path to the CSV file
     * @param rowLength Reference to store the number of columns (features + target)
     * @return A pair containing the flattened feature matrix and target vector
     */
    std::pair<std::vector<double>, std::vector<double>> readCSV(const std::string& filename, int& rowLength);
    
    /**
     * Writes prediction results to a file.
     * @param results The vector of prediction results to write
     * @param filename The path to the file to save the results
     */
    void writeResults(const std::vector<double>& results, const std::string& filename);
};

#endif // FUNCTIONS_IO_H
