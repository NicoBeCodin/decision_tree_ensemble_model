cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

# 添加子目录
add_subdirectory(functions_io)
add_subdirectory(functions_tree)
add_subdirectory(ensemble_bagging)

# 添加可执行文件
add_executable(MainEnsemble main.cpp)

# 链接库
target_link_libraries(MainEnsemble 
    FunctionsIO
    FunctionsTree
    Bagging
)

# 添加头文件路径
target_include_directories(MainEnsemble PRIVATE
    functions_io
    functions_tree
    ensemble_bagging
)
