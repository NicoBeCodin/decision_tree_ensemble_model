#include <iostream>
#include "functions_io.h"

int main() {
    // Define the name of the CSV file to read
    std::string filename = "../datasets/15k_random.csv";  // Adjust based on the actual file name

    // Read the entire content of the CSV file
    vector<vector<string>> csv_content = openCSV(filename);

    // If the file was read successfully, print the first 5 rows
    if (!csv_content.empty()) {
        cout << "Contents of the first 5 rows of the CSV file:" << endl;
        for (size_t i = 0; i < 5 && i < csv_content.size(); ++i) {
            for (size_t j = 0; j < csv_content[i].size(); ++j) {
                cout << csv_content[i][j] << " ";  // Print each cell in the row
            }
            cout << endl;  // Line break
        }
    } else {
        cout << "Unable to read the CSV file or the file is empty." << endl;
    }

    return 0;
}
