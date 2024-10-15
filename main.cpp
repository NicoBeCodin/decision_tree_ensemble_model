#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <variant>
#include <random>
#include <algorithm>

using namespace std;

typedef vector<vector<int>> Matrix;

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

vector<vector<string>> openCSVLimited(string fname, int n){
    vector<vector<string>> content;
    vector<string> row;
    string line, word;
    fstream file (fname, ios::in);
    int i =0;
    if (file.is_open()){
        while(getline(file, line)&& i<n){
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
        if (header[i] == column_name) return i;
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
Matrix processParametersCSV(vector<vector<string>>content)
{
    size_t column_number = content[0].size();
    size_t row_number = content.size();
    Matrix processed_parameters;
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


void printParamAndResults(vector<string> header,Matrix parameters, vector<float> results){
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
float calculateVariance(const vector<int>& result_values){
    if (result_values.empty()) {
        //This isn't an error case but still has to be noted
        printf("result_values empty, can't calculate average! returning 0.0\n");
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
int getMaxFeature(Matrix values, int feature_index){
    int max =0;
    for (int i =0; i<values.size(); ++i){
        if(values[i][feature_index]>max) max = values[i][feature_index];
    }
    return max;
}

int getMinFeature(Matrix values, int feature_index){
    int min =999999;
    for (int i =0; i<values.size(); ++i){
        if(values[i][feature_index]<min) min = values[i][feature_index];
    }
    return min;
}

float getMeanFeature(Matrix values, int feature_index){
    int mean =0;
    for (int i =0; i<values.size(); ++i){
        mean += values[i][feature_index];
    }
    float average = (float)mean / ((float)values.size()); 
    return average;
}

//Draw unique numbers to serve as indexes for rows in random sampling
vector<int> drawUniqueNumbers(int n, int rows){
    if (n > rows +1){
        printf("Row number is smaller than sample size, setting n =rows");
        n=rows; 
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




//compares the different best thresholds of each feature and returns the one m=with min weighted variance
Threshold compareThresholds(vector<Threshold> thresholds){
    Threshold best_threshold = thresholds[0]; 
    
    for (int i=1; i<thresholds.size(); ++i){
        if (thresholds[i].weighted_variance < best_threshold.weighted_variance){
            best_threshold = thresholds[i];
        }
    }
    return best_threshold;
}

//Find for a feature the best threshold by minimizing variance 
Threshold bestThresholdColumn(Matrix values, vector<float> results, int column_index){

    int best_threshold = 0;
    float min_weighted_variance = 999999.0;
    //For every value in of feature[i], use it as a threshold and see best score
    for (int i =0; i<values.size(); ++i){

        int threshold = values[i][column_index];
        vector<int> left;
        vector<int> right;

        for (int j = 0; j<values.size(); ++j){
            //Strictly inferior, could also be inferior, could replace by float (int+ 0.5) for better splitting (don't know)

            //We split the data into two subgroups and calculate the weighted variance
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

//Implement random sampling: instead of trying out all the different threshholds, sample for example 30 values and try them out as thresholds.
Threshold findBestSplitRandom(Matrix values, vector<float> results, int sample_size){
    
    //Generate our sample Matrix of size sample_size
    Matrix sample_tab;
    vector<float> sample_results;
    vector<int> tab_indexes = drawUniqueNumbers(sample_size, values.size());
    for (int index: tab_indexes){
        sample_tab.push_back(values[index]);
        sample_results.push_back(results[index]);
    }

    //Iterate through columns, finds best threshold for each column
    vector<Threshold> feature_threshold;

    for (int i =0; i<values[0].size(); ++i){
        feature_threshold.push_back(bestThresholdColumn(sample_tab, sample_results, i));
    }
    Threshold best_threshold = compareThresholds(feature_threshold);
    return best_threshold; 
}
//returns a list 
vector<int> splitOnThreshold(Threshold threshold, Matrix values){
    vector<int> goRight(values.size());
    for (int i =0; i<values.size(); ++i){
        if (values[i][threshold.feature_index] < threshold.value){
            goRight.push_back(0);
        }
        else{
            goRight.push_back(1);
        } 
    }
    return goRight;
}




//Create initial node with all the data that will then create the other ones
Node nodeInitiate(Matrix parameters, vector<float> results){
    Node initialNode;

    //Finds the best threshold
    //Sample size is defined as 30 but this has to be optimized, find a function with a good tradeoff between performance and accuracy
    Threshold  nodeThreshold= findBestSplitRandom(parameters, results, 30);
    initialNode.threshold = nodeThreshold;
    //We're building nodes not leaves (=final results of regression)
    initialNode.isLeaf = false;
    initialNode.nodeDepth = 1;

    //Perform the split
    Matrix leftValues;
    vector<float> leftResults;
    Matrix rightValues;
    vector<float> rightResults;

    //Get int vector that will tell which indexes go right or left
    vector<int> goRightIndex = splitOnThreshold(nodeThreshold, parameters);
    for (int i =0; i<parameters.size(); ++i){
        if (goRightIndex[i] == 0){
            leftValues.push_back(parameters[i]);
            leftResults.push_back(results[i]);
        }
        else {
            rightValues.push_back(parameters[i]);
            rightResults.push_back(results[i]);
        }
    }
    //CODE TO COMPLETE!!
    //Need to create the two subnodes with the splitted dataset by calling nodeBuilder function. Then have to put the two pointers in left and right

    return initialNode;

}

//this is a recursive function that should build two nodes from one parentNode
//There must be a max depth limit
Node nodeBuilder(Node parentNode){
    Node node;
    return node;
}   




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


    printf("\n trying to find best threshold on bigger dataset... \n");
    int sample_size = 30;
    Threshold test_best_threshold = findBestSplitRandom(parameters_values, results_values, sample_size); 
    printf("test_threshold feature_index: %d , value: %d, weighted_variance: %f \n",test_best_threshold.feature_index, test_best_threshold.value, test_best_threshold.weighted_variance);





    


    return 0;
}