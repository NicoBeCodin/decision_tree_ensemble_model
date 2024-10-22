#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <variant>
#include <random>
#include <algorithm>
#include <numeric>


using namespace std;

typedef vector<vector<int>> Matrix;

vector<vector<string>> openCSV(string fname);

vector<vector<string>> openCSVLimited(string fname, int n);

void printStringCSV(vector<vector<string>> content);

int getColumnIndex(vector<string> header, string column_name);

int convertToInt(const std::string& str);

float convertToFloat(const std::string& str);

Matrix processParametersCSV(vector<vector<string>>content);

vector<float> processResultsCSV(vector<vector<string>> content);

void printParamAndResults(vector<string> header, Matrix parameters, vector<float> results);

struct Threshold {
    int feature_index;
    int value;
    float weighted_variance;
};

//tree structure = leaves & nodes
struct Node {
    bool isLeaf;
    float value; //If leaf node, store predicted value (mean of target values)
    Threshold threshold; //Threshold to split on
    int nodeDepth; 
    Node* left; //Left child node
    Node* right; //Right child node
};

/*
Functions for tree decisions
*/

//calculate variance of array of ints 
float calculateVariance(const vector<int>& result_values);

int getMaxFeature(Matrix values, int feature_index);

int getMinFeature(Matrix values, int feature_index);

float getMeanFeature(Matrix values, int feature_index);

vector<int> drawUniqueNumbers(int n, int rows);

Threshold compareThresholds(vector<Threshold> thresholds);

Threshold bestThresholdColumn(Matrix values, vector<float> results, int column_index);

Threshold findBestSplitRandom(Matrix values, vector<float> results, int sample_size);

vector<int> splitOnThreshold(Threshold threshold, Matrix values);

Node nodeInitiate(Matrix parameters, vector<float> results);

Node nodeBuilder(Node parentNode, Matrix parameters, vector<float> results);


//print tree functions
void nodePrinter(Node node);

void treePrinter(Node tree);


#endif // DECISION_TREE_H