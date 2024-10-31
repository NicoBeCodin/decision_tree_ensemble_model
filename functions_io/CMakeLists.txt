cmake_minimum_required(VERSION 3.10)

# 工程名和版本
project(test_fonctions_io VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 启用编译器警告
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# 添加源文件
set(SOURCE_FILES test_functions_io.cpp functions_io.cpp)

# 添加可执行文件
add_executable(test_fonctions_io ${SOURCE_FILES})


