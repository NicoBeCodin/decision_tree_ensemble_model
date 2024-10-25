#include "functions_io.h"
#include "../functions_tree/functions_tree.h"

using namespace std;

typedef vector<vector<int>> Matrix;  // Define Matrix as an alias for a 2D integer vector

/* The openCSV function reads a csv file and transforms its *
 * lines into a vector then stores it in a global vector      */

vector<vector<string>> openCSV(string fname){
    vector<vector<string>> content;  // Store the contents of the CSV file
    vector<string> row;  // Store each row's data
    string line, word;  // Used to store the line and word being read
    fstream file (fname, ios::in);  // Open the file in read mode
    if (file.is_open()){
        while(getline(file, line)){
            row.clear();  // Clear the row vector

            stringstream str(line);  // Convert the line data into a string stream

            while(getline(str, word, ',')){  // Separate by comma
                row.push_back(word);  // Add the word to the current row
            }
            content.push_back(row);  // Add the row to the content
        }

    } else {
        // If unable to open the file
        cout << "Failed to open " << fname << endl;
    }
    return content;
}

/* The countCSVRows function counts the number of rows in a specified CSV file. *
 * It reads the file line by line and increments a counter for each line.       *
 * If the file cannot be opened, an error message is printed.                   */

int countCSVRows(const std::string& filePath) {
    std::ifstream file(filePath);
    std::string line;
    int rowCount = 0;

    // Open the file and read line by line to count rows
    if (file.is_open()) {
        while (std::getline(file, line)) {
            rowCount++;
        }
        file.close();
    } else {
        std::cerr << "Failed to open " << filePath << std::endl;
    }

    return rowCount;
}

/* Same as openCSV, with an additional condition for reading the lines (with a loop) *
 * , we take an additional parameter in input n, which will allow to stop the number * 
 * of lines taken. This function is useful for tests to not take too many values     */

vector<vector<string>> openCSVLimited(string fname, int n){
    vector<vector<string>> content;  // Store the contents of the CSV file
    vector<string> row;  // Store each row's data
    string line, word;  // Used to store the line and word being read
    fstream file (fname, ios::in);  // Open the file in read mode
    int i = 0;
    if (file.is_open()){
        while(getline(file, line) && i < n){  // Read up to the specified number of rows
            row.clear();  // Clear the row vector

            stringstream str(line);  // Convert the line data into a string stream

            while(getline(str, word, ',')){  // Separate by comma
                row.push_back(word);  // Add the word to the current row
            }
            content.push_back(row);  // Add the row to the content
            i++;  // Increase the row count
        }

    } else {
        // If unable to open the file
        cout << "Failed to open " << fname << endl;
    }
    return content;
}

/* The printStringCSV function, print the content of file when it's a vetor of string vector in input */

void printStringCSV(vector<vector<string>> content){
    int row_size = content.size();
    int col_size = content[0].size();

    for (size_t i = 0; i < row_size; ++i){  // Iterate over each row
        for (size_t j = 0; j < col_size; ++j){  // Iterate over each column
            cout << content[i][j] << " ";  // Print the current cell value
        }
        cout << "\n";  // New line
    }
}

/* the getColumnIndex function seeks to determine the index of a specific column in a vector. *
 * It returns the index if the column is found, and returns -1 if the column doesn't exist.   */

int getColumnIndex(vector<string> header, string column_name){
    for (size_t i = 0; i < header.size(); ++i){
        if (header[i] == column_name) return i;  // Find the index corresponding to the column name
    }
    return -1;  // Return -1 if not found
}

/* The convertToInt function converts a character string into an integer. If the conversion *
 * fails due to an invalid argument or an out-of-range value, it handles the error by       *
 * displaying an explanatory message and re-throwing the exception.                         */

int convertToInt(const std::string& str){
    try {
        return std::stoi(str);  // Convert to integer
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: Cannot convert '" << str << "' to int.\n";
        throw;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: Cannot convert '" << str << "' to int.\n";
        throw;
    }
}

/* The convertToFloat function is same as the convertToInt function but for a float */

float convertToFloat(const std::string& str) {
    try {
        return std::stof(str);  // Convert to float
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: Cannot convert '" << str << "' to float.\n";
        throw;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: Cannot convert '" << str << "' to float.\n";
        throw;
    }
}

/* The processParametersCSV function takes as input the contents of a CSV file in the        *
 * form of vectors of string vectors. It passes the first line, which is a header, and then  *
 * converts the integer values of each line (except for the last line, which concerns power, *
 * which must not be converted).                                                             */

Matrix processParametersCSV(vector<vector<string>> content){
    size_t column_number = content[0].size();  // Get the number of columns
    size_t row_number = content.size();  // Get the number of rows
    Matrix processed_parameters;  // Store the processed parameters
    // Skip the header row
    for (size_t i = 1; i < row_number; ++i){
        vector<int> processed_row;
        // Convert columns except the last one to integers
        for (size_t j = 0; j < column_number - 1; ++j){
            processed_row.push_back(convertToInt(content[i][j]));  // Convert to integer
        }
        processed_parameters.push_back(processed_row);  // Add the processed row
    }
    return processed_parameters;
}

/* The processResultsCSV function takes as input the contents of a CSV file in the 
 * form of vectors of string vectors. It skips the first line, which is a header, and then
 * converts the values in the last column of each line (assumed to be the result or target values)
 * to floating-point numbers, storing them in a vector to be returned. */

vector<float> processResultsCSV(vector<vector<string>> content){
    vector<float> processed_result;  // Store the processed results
    int result_column = content[0].size() - 1;  // Get the index of the result column
    // Skip the header row
    for (size_t i = 1; i < content.size(); ++i){
        processed_result.push_back(convertToFloat(content[i][result_column]));  // Convert to float and add to results
    }
    return processed_result;
}

/* The printParamAndResults function displays the headers, parameters and associated results of a data set in an array format */

void printParamAndResults(vector<string> header, Matrix parameters, vector<float> results){
    size_t column_number = header.size();  // Get the number of columns
    size_t row_number = parameters.size();  // Get the number of rows

    // Print header
    for (size_t k = 0; k < column_number; k++){
        cout << header[k] << " ";
    }
    printf("\n");

    // Print parameters and corresponding results
    for (size_t i = 0; i < row_number; ++i){
        for (size_t j = 0; j < column_number - 1; ++j){
            cout << parameters[i][j] << " ";  // Print each row's parameters
        }
        cout << results[i] << " " << endl;  // Print the result
    }
}

/* The nodePrinter function recursively prints details of a given node in the tree.                      *
 * For non-leaf nodes, it displays the address, depth, data size, threshold feature index,               *
 * threshold value, and weighted variance, then recursively calls itself on the left and right children. *
 * For leaf nodes, it displays the address, depth, data size, and mean value.                            */
void nodePrinter(Node* node){
    if (!node->isLeaf){
        printf("Node address: ");
        for (auto i : node->address) printf("%d", i);
        printf("\n");
        printf("Node depth: %d\nData size: %d\nThreshold feature_index: %d\nThreshold value: %d\nWeighted variance: %f\n\n", node->nodeDepth, node->data_size, node->threshold.feature_index, node->threshold.value, node->threshold.weighted_variance);
        nodePrinter(node->left);
        nodePrinter(node->right);        

    }
    else {
        printf("\nLeaf address: ");
        for (auto i : node->address) printf("%d", i);
        printf("\n");
        printf("Leaf depth: %d\nData size: %d\nMean value: %f\n", node->nodeDepth, node->data_size, node->value);
    }
}

/* The treePrinter function serves as the main entry point for printing the structure        *
 * and details of the entire tree. It starts by printing a header and then calls nodePrinter *
 * on the root node, which recursively traverses and prints each node in the tree.           */

void treePrinter(Node* root){
    printf("Printing tree...\n");
    nodePrinter(root);
}