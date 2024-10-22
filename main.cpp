#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <variant>
#include <random>
#include <algorithm>

#include "functions.h"


int main(){
    //Open a cert
    int number_of_rows = 500;
    vector<vector<string>> content = openCSVLimited("datasets/15k_ga_adaptive.csv", number_of_rows);

    //Get header
    vector<string> string_header = content[0]; 
    Matrix parameters_values = processParametersCSV(content);
    vector<float> results_values = processResultsCSV(content);
    
    //This makes sure the two outputs are the same and and parameters_values and results_values are rightfully processed
    //printStringCSV(content);
    printf("\n");
    //printParamAndResults(string_header, parameters_values, results_values);


    /*Test bestThresholdColumn
    Find the best point on which to split the dataset
    */
    printf("\n Testing out on handwritten dataset...");
    Matrix test_tab = {{1,2},{2,2},{3,2},{6,2},{4,2},{5,2}};
    vector<float> test_results = {0.1, 0.2,0.3,0.6,0.4,0.5};
    int test_column =0;
    Threshold test_thresh = bestThresholdColumn(test_tab, test_results, test_column);
    printf("test_threshold feature_index: %d , value: %d, weighted_variance: %f \n",test_thresh.feature_index, test_thresh.value, test_thresh.weighted_variance);


    printf("\nTrying to find best threshold on bigger dataset... \n");
    int sample_size = 30;
    Threshold test_best_threshold = findBestSplitRandom(parameters_values, results_values, sample_size); 
    printf("test_threshold feature_index: %d , value: %d, weighted_variance: %f \n",test_best_threshold.feature_index, test_best_threshold.value, test_best_threshold.weighted_variance);


    printf("\nTrying to build tree...\n");
    Node tree = nodeInitiate(parameters_values, results_values);
    printf("\nTrying to print tree... \n");

    treePrinter(tree);

    return 0;
}