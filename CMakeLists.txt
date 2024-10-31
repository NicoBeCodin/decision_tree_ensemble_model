# CMake 最小版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(MyMultiProgramProject CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# 添加子目录 
add_subdirectory(data_clean)
add_subdirectory(functions_io)

