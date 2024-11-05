# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Define project name and project language
project(RegressionTreeProject LANGUAGES CXX)

# Set C++ standard to C++11 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source and header file paths
add_subdirectory(functions_io)
add_subdirectory(functions_tree)

# Specify main.cpp as the main program source file
add_executable(RegressionTreeApp main.cpp)

# Link functions_io and functions_tree to the main program
target_link_libraries(RegressionTreeApp PRIVATE FunctionsIO RegressionTree)
