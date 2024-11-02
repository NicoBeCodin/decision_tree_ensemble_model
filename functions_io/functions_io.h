#ifndef FUNCTIONS_IO_H
#define FUNCTIONS_IO_H

#include <vector>
#include <string>
#include <utility>
/* Code Explanation

    readCSV method: Reads data from a CSV file, treating the last value in each row as the target variable 'performance', and the remaining values as features.
        Uses std::ifstream to open the CSV file.
        Skips the first line, assuming it is a header.
        Parses each line's data as type double, storing it in a feature matrix and target vector.
        Returns a std::pair, where the first element is the feature matrix and the second element is the target vector.

    writeResults method: Writes prediction results to a specified file, with each result on a new line.
        Uses std::ofstream to open the output file.
        Writes each prediction result line by line. */

class DataIO {
public:
    /**
     * Reads data from a CSV file.
     * @param filename The path to the CSV file
     * @return A pair containing the feature matrix and target vector; the first element is the feature matrix, and the second element is the target vector
     */
    std::pair<std::vector<std::vector<double>>, std::vector<double>> readCSV(const std::string& filename);
    
    /**
     * Writes prediction results to a file.
     * @param results The vector of prediction results to write
     * @param filename The path to the file to save the results
     */
    void writeResults(const std::vector<double>& results, const std::string& filename);
};

#endif // FUNCTIONS_IO_H
