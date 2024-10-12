#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <variant>

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