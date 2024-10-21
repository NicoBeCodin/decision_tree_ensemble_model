#include "functions_io.h"
#include <iostream>

int main() {
    // 定义要读取的CSV文件名
    std::string filename = "../datasets/15k_random.csv";  // 请根据实际文件名修改

    // 读取CSV文件的全部内容
    vector<vector<string>> csv_content = openCSV(filename);

    // 如果文件读取成功，打印前5行
    if (!csv_content.empty()) {
        cout << "CSV文件的前5行内容：" << endl;
        for (size_t i = 0; i < 5 && i < csv_content.size(); ++i) {
            for (size_t j = 0; j < csv_content[i].size(); ++j) {
                cout << csv_content[i][j] << " ";  // 打印每一行的单元格
            }
            cout << endl;  // 换行
        }
    } else {
        cout << "无法读取CSV文件或文件为空。" << endl;
    }

    return 0;
}
