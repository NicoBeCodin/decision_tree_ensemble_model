#include "functions_tree.h"
#include <iostream>  // Include the iostream header to use output functions from the standard library
#include <vector>    // Include the vector header
#include <random>    // Include the random header
#include <algorithm> // Include the algorithm header to use std::shuffle

using namespace std;



/* The calculateVariance function computes the variance of an array of integers. *
 * It first checks if the array is empty, returning 0.0 if it is.                *
 * Then, it calculates the mean of the values and determines the variance        *
 * based on that mean. The function returns the calculated variance.             *
 * It could be adjusted to provide an unbiased estimate of variance.             */
 
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

/* The getMaxFeature function finds the maximum value of a specific feature  *
 * across all rows in a given matrix. It iterates through each row, checking *
 * the specified feature index and updating the maximum value accordingly.   *
 * The function returns the highest value found for the specified feature.   */

int getMaxFeature(Matrix& values, int feature_index){
    // Check if matrix is empty
    if (values.empty() || values[0].empty()) {
        throw std::out_of_range("Matrix is empty");
    }

    // Checks whether the characteristic index is outside the limits
    if (feature_index < 0 || feature_index >= (int)values[0].size()) {
        throw std::out_of_range("Feature index is out of bounds");
    }

    int max = values[0][feature_index];
    int row_size = values.size();

    for (int i = 0; i < row_size; ++i){
        if(values[i][feature_index]>max) max = values[i][feature_index];
    }
    return max;
}

/* The getMinFeature function finds the minimum value of a specific feature  *
 * across all rows in a given matrix. It iterates through each row, checking *
 * the specified feature index and updating the minimum value accordingly.   *
 * The function returns the lowest value found for the specified feature.    */

int getMinFeature(Matrix& values, int feature_index){
    // Check if matrix is empty
    if (values.empty() || values[0].empty()) {
        throw std::out_of_range("Matrix is empty");
    }

    // Checks whether the characteristic index is outside the limits
    if (feature_index < 0 || feature_index >= (int)values[0].size()) {
        throw std::out_of_range("Feature index is out of bounds");
    }

    int min = values[0][feature_index];
    int row_size = values.size();
    for (int i = 0; i < row_size; ++i){
        if(values[i][feature_index]<min) min = values[i][feature_index];
    }
    return min;
}

/* The getMeanFeature function calculates the mean value of a specific feature  *
 * across all rows in a given matrix. It iterates through each row, summing the *
 * values of the specified feature index. The average is then computed by       *
 * dividing the total sum by the number of rows. The function returns the       *
 * calculated mean value for the specified feature.                             */

float getMeanFeature(Matrix& values, int feature_index){
    // Check if matrix is empty
    if (values.empty() || values[0].empty()) {
        throw std::out_of_range("Matrix is empty");
    }

    // Checks whether the characteristic index is outside the limits
    if (feature_index < 0 || feature_index >= (int)values[0].size()) {
        throw std::out_of_range("Feature index is out of bounds");
    }

    int mean = 0;
    int row_size = values.size();

    for (int i = 0; i < row_size; ++i){
        mean += values[i][feature_index];
    }
    float average = (float)mean / ((float)values.size()); 
    return average;
}


/* The drawUniqueNumbers function generates a vector of unique indices for        *
 * random sampling from a given number of rows. If the sample size (n) is         *
 * greater than the total number of rows, it adjusts n to be equal to rows and    *
 * returns a simple incremented array. Otherwise, it initializes a vector with    *
 * row indices, shuffles them randomly, and selects the first n unique indices.   *
 * The function returns a vector containing the randomly selected unique indices. */

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


/* The compareThresholds function evaluates the best thresholds from a vector    *
 * of Threshold objects and returns the one with the minimum weighted variance.  *
 * It initializes the best_threshold to the first element and iterates through   *
 * the remaining thresholds, updating the best_threshold whenever it finds a     *
 * threshold with a lower weighted variance. The function ultimately returns the *
 * threshold that minimizes the weighted variance.                               */

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

/* The bestThresholdColumn function determines the optimal threshold for a       *
 * specified feature in order to minimize the variance. It iterates through each *
 * value of the feature as a potential threshold, splitting the data into two    *
 * subgroups based on this threshold. For each split, it calculates the weighted *
 * variance of the left and right subgroups. The function keeps track of the     *
 * threshold that results in the lowest weighted variance. Finally, it returns a *
 * Threshold object containing the column index, the best threshold, and the     *
 * minimum weighted variance found.                                              */

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

/* The findBestSplitRandom function implements random sampling to determine the    *
 * optimal threshold for splitting data in a given feature. Instead of evaluating  *
 * all possible thresholds, it randomly samples a specified number of values from  *
 * the dataset to use as potential thresholds. The function first generates a      *
 * sample matrix and corresponding results based on unique random indexes. It then *
 * iterates through each feature column, determining the best threshold for each   *
 * column using the sampled data. Finally, it compares the thresholds across all   *
 * features and returns the threshold with the lowest weighted variance.           */

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

/* The splitOnThreshold function takes a Threshold object and a matrix of values as   *
 * inputs. It iterates through each row of the matrix, checking the specified feature *
 * index against the threshold value. For each row, it appends a value to the output  *
 * vector: a 0 if the feature value is less than the threshold, and a 1 if it is      *
 * greater than or equal to the threshold. The function returns a vector indicating   *
 * the direction (left or right) for each row based on the threshold.                 */

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


/* The nodeInitiate function creates an initial node for a decision tree using a matrix of    *
 * parameters and a vector of results. It first finds the best threshold for splitting the    *
 * data by calling the findBestSplitRandom function with a sample size of 30, which should    *
 * be optimized for performance and accuracy. The function initializes the node's properties, *
 * such as depth and data size, and then splits the data into left and right subsets based    *
 * on the selected threshold. It subsequently calls the nodeBuilder function to create        *
 * subnodes for the left and right splits, assigning them to the initial node's left and      *
 * right pointers, respectively. Finally, it returns the initialized node.                    */

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

/* The nodeBuilder function is a recursive function that constructs child nodes from a given *
 * parent node in a decision tree. It first checks if the maximum depth of the tree has      *
 * been reached; if so, it creates a leaf node with the mean of the results as its value.    *
 * If the depth is within limits, it calculates the best threshold for splitting the data    *
 * using the findBestSplitRandom function. The data is then divided into left and right      *
 * subsets based on this threshold. The function then recursively calls itself to build      *
 * left and right child nodes, linking them to the current node. Finally, it returns the     *
 * newly created node, which could either be a leaf or a decision node.                      */

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

