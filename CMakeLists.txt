cmake_minimum_required(VERSION 3.10)
project(DecisionTreeSingleModel)

set(CMAKE_CXX_STANDARD 17)

# 添加子目录
add_subdirectory(functions_io)
add_subdirectory(functions_tree)

# 添加可执行文件
add_executable(MainDecisionTree main.cpp)

# 链接库
target_link_libraries(MainDecisionTree FunctionsIO FunctionsTree)

# 添加头文件路径
target_include_directories(MainDecisionTree PRIVATE
    functions_io
    functions_tree
)
