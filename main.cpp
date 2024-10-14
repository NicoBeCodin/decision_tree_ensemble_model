#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <variant>
#include <random>
#include <algorithm>

using namespace std;



vector<vector<string>> openCSV(string fname){
    vector<vector<string>> content;
    vector<string> row;
    string line, word;
    fstream file (fname, ios::in);
    if (file.is_open()){
        while(getline(file, line)){
            row.clear();

            stringstream str(line);

            while(getline(str, word, ',')){
                row.push_back(word);
            }
            content.push_back(row);
        }

    }
    else{
        //If it can't open csv file
        cout<<"Failed to open " << fname <<endl;
    }
    return content;
}

//Print the file when it's a string
void printStringCSV(vector<vector<string>> content){
        for (int i = 0; i<content.size();++i){
        for (int j=0; j<content[i].size();++j){
            cout<<content[i][j]<<" ";
        }
        cout<<"\n";
    }
}

int getColumnIndex(vector<string> header, string column_name){
    for (int i =0; i<header.size(); ++i){
        if (header[i] == "column_name") return i;
    }
    return -1;
}


//These converting functions are not "no-value" proof, this could be added
int convertToInt(const std::string& str){

    try {
        return std::stoi(str);  // Convert to int
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: Cannot convert '" << str << "' to int.\n";
        throw;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: Cannot convert '" << str << "' to int.\n";
        throw;
    }
}

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


//These two processing functions assumme the result is in last column
//If this turns out not to be necessarly the case, getColumnIndex can be implemented
vector<vector<int>> processParametersCSV(vector<vector<string>>content)
{
    size_t column_number = content[0].size();
    size_t row_number = content.size();
    vector<vector<int>> processed_parameters;
    //Skip first row (header)
    for (int i = 1; i<row_number; ++i){
        vector<int> processed_row;
        //Don't convert last row to int (performance is a float)
        for (int j =0; j<column_number- 1; ++j){
            processed_row.push_back(convertToInt(content[i][j]));
        }
        processed_parameters.push_back(processed_row);
    }
    return processed_parameters;
}

vector<float> processResultsCSV(vector<vector<string>> content){
    vector<float> processed_result;
    int result_column = content[0].size()-1;
    //Skip first row (header)
    for (int i =1; i<content.size(); ++i){
        processed_result.push_back(convertToFloat(content[i][result_column]));
    }
    return processed_result;
}


void printParamAndResults(vector<string> header,vector<vector<int>> parameters, vector<float> results){
    size_t column_number = header.size();
    size_t row_number = parameters.size();

    for (int k = 0; k<column_number; k++){
        cout<<header[k]<<" ";
    }
    printf("\n");

    for (int i=0; i<row_number; ++i){
        for (int j=0; j<column_number -1; ++j){
            cout<<parameters[i][j]<<" ";
        }
        cout<<results[i]<<" "<< endl;
    }

}

/* DECISION TREE LOGIC



*/
//tree structure = leaves & nodes
struct Node {
    bool isLeaf;
    float value; //If leaf node, store predicted value (mean of target values)
    int featureIndex; //Index of feature to split on 
    float threshold; //Threshold to split on
    Node* left; //Left child node
    Node* right; //Right child node

    //Default
    Node(): isLeaf(true), value(0.0), featureIndex(-1), threshold(0.0), left(nullptr), right(nullptr) {}

};



/*
Functions for tree decisions
*/

//calculate variance of array of ints 
float calculateVariance(const vector<int>& result_values){
    if (result_values.empty()) {
        printf("result_values empty, can't calculate average! returning 0.0");
        return 0.0;
    }
    float mean=0.0;
    for (float val: result_values) {
        mean+=val;
    }
    mean /= (float)result_values.size();

    float variance =0.0;
    for (int val: result_values){
        variance += ((float)val - mean) * ((float)val -mean );
    }
    //Could return variance / (result_values.size() -1) for unbiased estimator;
    return variance / (float)result_values.size();

}

//instead of trying out all values, get the min, max and mean of a feature list and work from there
int getMaxFeature(vector<vector<int>> values, int feature_index){
    int max =0;
    for (int i =0; i<values.size(); ++i){
        if(values[i][feature_index]>max) max = values[i][feature_index];
    }
    return max;
}

int getMinFeature(vector<vector<int>> values, int feature_index){
    int min =999999;
    for (int i =0; i<values.size(); ++i){
        if(values[i][feature_index]<min) min = values[i][feature_index];
    }
    return min;
}

float getMeanFeature(vector<vector<int>> values, int feature_index){
    int mean =0;
    for (int i =0; i<values.size(); ++i){
        mean += values[i][feature_index];
    }
    float average = (float)mean / ((float)values.size()); 
    return average;
}

//Implement random sampling: instead of trying out all the different threshholds, sample for example 30 values and try them out as thresholds.

//findBestSplitRandom(vector<vector<int>> values, vector<float> results, int sample_size)


/*For specific feature, find the best threshhold
    
*/

struct Threshold {
    int feature_index;
    int value;
    float weighted_variance;

};

Threshold bestThresholdColumn(vector<vector<int>> values, vector<float> results, int column_index){

    int best_threshold = 0;
    float min_weighted_variance = 999999.0;
    //For every value in of feature[i], use it as a threshold and see best score
    for (int i =0; i<values.size(); ++i){

        int threshold = values[i][column_index];
        vector<int> left;
        vector<int> right;

        for (int j = 0; j<values.size(); ++j){
            //Strictly inferior, could also be inferior, could replace by float (int+ 0.5) for better splitting (don't know)
            if (values[j][column_index] < threshold ){
                left.push_back(values[j][column_index]);
            } else {
                right.push_back(values[j][column_index]);
            }

        }
        float weighted_variance = (calculateVariance(left)* (float)left.size() + calculateVariance(right) * (float)right.size())/values.size();
        if (weighted_variance < min_weighted_variance){
            min_weighted_variance = weighted_variance;
            best_threshold = threshold;
        }
    }

    return Threshold{column_index, best_threshold, min_weighted_variance};
}

//Draw unique numbers to serve as indexes for rows in random sampling
vector<int> drawUniqueNumbers(int n, int rows){
    if (n > rows +1){
        throw invalid_argument("rows < number of unique numbers!");
    }

    vector<int> numbers(rows);
    for (int i =0; i<rows; ++i){
        numbers[i] = i;
    }
    random_device rd;
    mt19937 g(rd());

    shuffle(numbers.begin(), numbers.end(), g);
    vector<int> result (numbers.begin(),numbers.begin()+n );
    return result;

}


int main(){

    vector<vector<string>> content = openCSV("datasets/test.csv");

    //Get header
    vector<string> string_header = content[0]; 
    vector<vector<int>> parameters_values = processParametersCSV(content);
    vector<float> results_values = processResultsCSV(content);
    
    //This makes sure the two outputs are the same and and parameters_values and results_values are rightfully processed
    printStringCSV(content);
    printf("\n");
    printParamAndResults(string_header, parameters_values, results_values);

    


    return 0;
}