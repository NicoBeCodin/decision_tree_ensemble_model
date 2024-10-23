#include "functions_tree.h"
#include <iostream>  // 引入iostream头文件以使用标准库中的输出函数
#include <vector>    // 引入vector头文件
#include <random>    // 引入random头文件
#include <algorithm> // 引入algorithm头文件以使用std::shuffle

using namespace std;



//calculate variance of array of ints 
float calculateVariance(const vector<int>& result_values){
    if (result_values.empty()) {
        //This isn't an error case but still has to be noted
        //printf("result_values empty, can't calculate average! returning 0.0\n");
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
    int row_size = values.size();
    for (int i =0; i<row_size; ++i){
        if(values[i][feature_index]>max) max = values[i][feature_index];
    }
    return max;
}

int getMinFeature(Matrix values, int feature_index){
    int min =999999;
    int row_size = values.size();
    for (int i =0; i<row_size; ++i){
        if(values[i][feature_index]<min) min = values[i][feature_index];
    }
    return min;
}

float getMeanFeature(Matrix values, int feature_index){
    int mean =0;
    int row_size = values.size();
    for (int i =0; i<row_size; ++i){
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
        //return just a simple incremented array, no need for randomness
        vector<int> vec(n);
        generate(vec.begin(), vec.end(), [] {
        static int i = 0;
        return i++;
        });
        return vec;
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
Threshold compareThresholds(vector<Threshold>& thresholds){
    Threshold best_threshold = thresholds[0]; 
    int thresholds_size = thresholds.size();
    
    for (int i=1; i<thresholds_size; ++i){
        if (thresholds[i].weighted_variance < best_threshold.weighted_variance){
            best_threshold = thresholds[i];
        }
    }
    return best_threshold;
}

//Find for a feature the best threshold by minimizing variance 
Threshold bestThresholdColumn(Matrix& values, vector<float>& results, int column_index){

    int best_threshold = 0;
    float min_weighted_variance = 999999.0;
    int row_size = values.size();
    //For every value in of feature[i], use it as a threshold and see best score
    for (int i =0; i<row_size; ++i){

        int threshold = values[i][column_index];
        vector<int> left;
        vector<int> right;

        for (int j = 0; j<row_size; ++j){
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

    return Threshold(column_index, best_threshold, min_weighted_variance);
}

//Implement random sampling: instead of trying out all the different threshholds, sample for example 30 values and try them out as thresholds.
Threshold findBestSplitRandom(Matrix& values, vector<float>& results, int sample_size){
    
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
    int col_size = values[0].size();
    for (int i =0; i<col_size; ++i){
        feature_threshold.push_back(bestThresholdColumn(sample_tab, sample_results, i));
    }
    Threshold best_threshold = compareThresholds(feature_threshold);
    return best_threshold; 
}

//returns a list 
vector<int> splitOnThreshold(Threshold& threshold, Matrix& values){
    vector<int> goRight;
    int row_size = values.size();
    for (int i =0; i<row_size; ++i){
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
Node* nodeInitiate(Matrix& parameters, vector<float>& results){
    Node* initialNode = new Node();

    //Finds the best threshold
    //Sample size is defined as 30 but this has to be optimized, find a function with a good tradeoff between performance and accuracy
    Threshold nodeThreshold= findBestSplitRandom(parameters, results, 30);
    initialNode->threshold = nodeThreshold;
    //We're building nodes not leaves (=final results of regression)
    initialNode->isLeaf = false;
    initialNode->nodeDepth = 1;
    initialNode->data_size = (int)parameters.size();

    //Perform the split
    Matrix leftValues;
    vector<float> leftResults;
    Matrix rightValues;
    vector<float> rightResults;

    //Get int vector that will tell which indexes go right or left
    vector<int> goRightIndex = splitOnThreshold(nodeThreshold, parameters);
    int row_size = parameters.size();
    for (int i =0; i<row_size; ++i){
        if (goRightIndex[i] == 0){
            leftValues.push_back(parameters[i]);
            leftResults.push_back(results[i]);
        }
        else {
            rightValues.push_back(parameters[i]);
            rightResults.push_back(results[i]);
        }
    }
    printf("nodeInitiate split finished...\n");


    //CODE TO COMPLETE!!
    //Need to create the two subnodes with the splitted dataset by calling nodeBuilder function. Then have to put the two pointers in left and right

    //create an adress code for each node
    Node* leftNode = nodeBuilder(initialNode, leftValues, leftResults, false);

    initialNode->left = leftNode;
    printf("initialNode.left process finished...\n");

    Node* rightNode = nodeBuilder(initialNode, rightValues, rightResults, true);

    initialNode->right = rightNode;
    printf("initialNode.right process finished...\n");

    return initialNode;
}

//this is a recursive function that should build two nodes from one parentNode
//node builder is the same as nodeInitiate except it needs a parentNode
Node* nodeBuilder(Node* parentNode, Matrix& parameters, vector<float>& results, bool right){
    //Break case if depth is too big
    int max_depth = 3;
    Node* currentNode = new Node();
    currentNode->data_size = (int)parameters.size();

    int r = right ? 1: 0; 
    currentNode->adress = parentNode->adress; 
    currentNode->adress.push_back(r);
    
    printf("Node has %d values to start\n", (int)parameters.size());
    if (parentNode->nodeDepth > max_depth){
        //return a leaf = end of tree
        currentNode->isLeaf =true;
        currentNode->nodeDepth = parentNode->nodeDepth +1;
        float mean = accumulate(results.begin(), results.end(), 0) / (float)results.size();
        currentNode->value = mean;
        printf("Leaf node created with mean: %f\n", mean);
    } else {
        //General case
        currentNode->isLeaf = false;
        currentNode->nodeDepth = parentNode->nodeDepth + 1;
        Threshold nodeThreshold = findBestSplitRandom(parameters, results, 30);
        currentNode->threshold = nodeThreshold;
        printf("Threshold calculated to be on feature %d, value: %d, weighted_var: %f\n", nodeThreshold.feature_index, nodeThreshold.value, nodeThreshold.weighted_variance);

        Matrix leftValues;
        vector<float> leftResults;
        Matrix rightValues;
        vector<float> rightResults;

        vector<int> goRightIndex = splitOnThreshold(nodeThreshold, parameters);
        int row_size = parameters.size();
        for (int i =0; i<row_size; ++i){
        if (goRightIndex[i] == 0){
            leftValues.push_back(parameters[i]);
            leftResults.push_back(results[i]);
        }
        else {
            rightValues.push_back(parameters[i]);
            rightResults.push_back(results[i]);
        }
        }
        printf("nodeBuilder splitting finished...\n");
        printf("Left group size: %d, Right group size %d\n", (int)leftValues.size(), (int)rightValues.size());

        Node* leftNode = nodeBuilder(currentNode, leftValues, leftResults, false);
        currentNode->left = leftNode;

        Node* rightNode = nodeBuilder(currentNode, rightValues, rightResults, true);
        currentNode->right = rightNode;
        
        printf("nodeBuilder process finished...\n");
    }

    printf("Returning currentNode\n");
    return currentNode;
}  

