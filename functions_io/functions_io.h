#ifndef FUNCTIONS_IO_H
#define FUNCTIONS_IO_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

// Early declaration of Node to avoid circular dependencies
struct Node;

// Define Matrix as an alias for a 2D integer vector
typedef vector<vector<int>> Matrix;

// Function declarations
vector<vector<string>> openCSV(string fname);  // Open a CSV file and read its contents
int countCSVRows(const std::string& filePath);  // Return the number of rows in a CSV file
vector<vector<string>> openCSVLimited(string fname, int n);  // Read a specified number of rows from a CSV file
void printStringCSV(vector<vector<string>> content);  // Print CSV file contents

int getColumnIndex(vector<string> header, string column_name);  // Get the index of a column by name
int convertToInt(const std::string& str);  // Convert a string to an integer
float convertToFloat(const std::string& str);  // Convert a string to a float

Matrix processParametersCSV(vector<vector<string>> content);  // Process the parameter part of the CSV file
vector<float> processResultsCSV(vector<vector<string>> content);  // Process the result part of the CSV file

void printParamAndResults(vector<string> header, Matrix parameters, vector<float> results);  // Print parameters and results

// Print tree functions
void nodePrinter(Node* node);

void treePrinter(Node* node);

#endif // FUNCTIONS_IO_H
