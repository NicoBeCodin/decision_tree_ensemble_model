#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

// 定义Matrix为二维整数向量的别名
typedef vector<vector<int>> Matrix;

// 函数声明
vector<vector<string>> openCSV(string fname);  // 打开CSV文件并读取内容
int countCSVRows(const std::string& filePath);  // 返回CSV文件的行数
vector<vector<string>> openCSVLimited(string fname, int n);  // 读取指定行数的CSV数据
void printStringCSV(vector<vector<string>> content);  // 打印CSV文件内容

int getColumnIndex(vector<string> header, string column_name);  // 获取列名索引
int convertToInt(const std::string& str);  // 将字符串转换为整数
float convertToFloat(const std::string& str);  // 将字符串转换为浮点数

Matrix processParametersCSV(vector<vector<string>> content);  // 处理CSV参数部分
vector<float> processResultsCSV(vector<vector<string>> content);  // 处理CSV结果部分

void printParamAndResults(vector<string> header, Matrix parameters, vector<float> results);  // 打印参数和结果

#endif // FUNCTIONS_H
