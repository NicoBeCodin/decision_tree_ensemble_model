#include "functions_io.h"
#include "functions_tree.h"



using namespace std;

typedef vector<vector<int>> Matrix;  // 定义Matrix为二维整数向量的别名

// 打开CSV文件并读取内容，返回一个二维字符串向量
vector<vector<string>> openCSV(string fname){
    vector<vector<string>> content;  // 存储CSV文件的内容
    vector<string> row;  // 存储每一行的数据
    string line, word;  // 用于存储读取的行和单词
    fstream file (fname, ios::in);  // 打开文件以读取模式
    if (file.is_open()){
        while(getline(file, line)){
            row.clear();  // 清空行向量

            stringstream str(line);  // 将行数据转换为字符串流

            while(getline(str, word, ',')){  // 按逗号分隔
                row.push_back(word);  // 将单词添加到当前行
            }
            content.push_back(row);  // 将行添加到内容中
        }

    } else {
        // 如果无法打开文件
        cout<<"Failed to open " << fname <<endl;
    }
    return content;
}

// 函数：返回CSV文件的行数
int countCSVRows(const std::string& filePath) {
    std::ifstream file(filePath);
    std::string line;
    int rowCount = 0;

    // 打开文件并逐行读取，统计行数
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

// 打开CSV文件并读取指定行数的数据
vector<vector<string>> openCSVLimited(string fname, int n){
    vector<vector<string>> content;  // 存储CSV文件的内容
    vector<string> row;  // 存储每一行的数据
    string line, word;  // 用于存储读取的行和单词
    fstream file (fname, ios::in);  // 打开文件以读取模式
    int i = 0;
    if (file.is_open()){
        while(getline(file, line) && i < n){  // 读取指定数量的行
            row.clear();  // 清空行向量

            stringstream str(line);  // 将行数据转换为字符串流

            while(getline(str, word, ',')){  // 按逗号分隔
                row.push_back(word);  // 将单词添加到当前行
            }
            content.push_back(row);  // 将行添加到内容中
            i++;  // 增加行计数
        }

    } else {
        // 如果无法打开文件
        cout<<"Failed to open " << fname <<endl;
    }
    return content;
}

// 打印CSV文件内容
void printStringCSV(vector<vector<string>> content){
    int row_size = content.size();
    int col_size = content[0].size();

    for (size_t i = 0; i < row_size; ++i){  // 遍历每一行
        for (size_t j = 0; j < col_size; ++j){  // 遍历每一列
            cout<<content[i][j]<<" ";  // 打印当前单元格的值
        }
        cout<<"\n";  // 换行
    }
}

// 获取列名在表头中的索引
int getColumnIndex(vector<string> header, string column_name){
    for (size_t i = 0; i < header.size(); ++i){
        if (header[i] == column_name) return i;  // 找到列名对应的索引
    }
    return -1;  // 如果未找到返回-1
}

// 将字符串转换为整数
int convertToInt(const std::string& str){
    try {
        return std::stoi(str);  // 转换为整数
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: Cannot convert '" << str << "' to int.\n";
        throw;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: Cannot convert '" << str << "' to int.\n";
        throw;
    }
}

// 将字符串转换为浮点数
float convertToFloat(const std::string& str) {
    try {
        return std::stof(str);  // 转换为浮点数
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: Cannot convert '" << str << "' to float.\n";
        throw;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: Cannot convert '" << str << "' to float.\n";
        throw;
    }
}

// 处理CSV文件的参数部分，假设结果列是最后一列，返回矩阵
Matrix processParametersCSV(vector<vector<string>> content){
    size_t column_number = content[0].size();  // 获取列数
    size_t row_number = content.size();  // 获取行数
    Matrix processed_parameters;  // 存储处理后的参数
    // 跳过表头行
    for (size_t i = 1; i < row_number; ++i){
        vector<int> processed_row;
        // 将除最后一列外的列转换为整数
        for (size_t j = 0; j < column_number - 1; ++j){
            processed_row.push_back(convertToInt(content[i][j]));  // 转换为整数
        }
        processed_parameters.push_back(processed_row);  // 添加处理后的行
    }
    return processed_parameters;
}

// 处理CSV文件的结果部分，假设结果在最后一列，返回浮点数向量
vector<float> processResultsCSV(vector<vector<string>> content){
    vector<float> processed_result;  // 存储处理后的结果
    int result_column = content[0].size() - 1;  // 获取结果列的索引
    // 跳过表头行
    for (size_t i = 1; i < content.size(); ++i){
        processed_result.push_back(convertToFloat(content[i][result_column]));  // 转换为浮点数并添加到结果
    }
    return processed_result;
}

// 打印参数和结果
void printParamAndResults(vector<string> header, Matrix parameters, vector<float> results){
    size_t column_number = header.size();  // 获取列数
    size_t row_number = parameters.size();  // 获取行数

    // 打印表头
    for (size_t k = 0; k < column_number; k++){
        cout << header[k] << " ";
    }
    printf("\n");

    // 打印参数和对应的结果
    for (size_t i = 0; i < row_number; ++i){
        for (size_t j = 0; j < column_number - 1; ++j){
            cout << parameters[i][j] << " ";  // 打印每行参数
        }
        cout << results[i] << " " << endl;  // 打印结果
    }
}

void nodePrinter(Node* node){
    if (!node->isLeaf){
        printf("node adress: ");
        for (auto i : node->adress) printf("%d", i);
        printf("\n");
        printf("node depth: %d\ndata size: %d\nthreshold feature_index: %d\nthreshold value %d\nweighted_variance %f\n\n", node->nodeDepth, node->data_size, node->threshold.feature_index, node->threshold.value, node->threshold.weighted_variance);
        nodePrinter(node->left);
        nodePrinter(node->right);        

    }
    else {
        printf("\nleaf adress: ");
        for (auto i : node->adress) printf("%d", i);
        printf("\n");
        printf("leaf depth: %d\ndata size: %d \nmean value: %f\n", node->nodeDepth, node->data_size, node->value);
    }

}

//print the tree structure and it's values
void treePrinter(Node* root){
    printf("Printing tree...\n");
    nodePrinter(root);
}


