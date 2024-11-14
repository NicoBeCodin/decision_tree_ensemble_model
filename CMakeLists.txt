# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Define project name and project language
project(DecisionTreeEnsembleModel LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add subdirectories for modules
add_subdirectory(functions_io)
add_subdirectory(functions_tree)
add_subdirectory(ensemble_bagging)
add_subdirectory(ensemble_boosting)

# Specify main.cpp as the main program source file
add_executable(MainEnsemble main.cpp)

# Link all necessary libraries to the main executable
target_link_libraries(MainEnsemble PUBLIC FunctionsIO FunctionsTree Bagging Boosting)

# Include directories for the main executable
target_include_directories(MainEnsemble PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_io
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_tree
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_boosting
)
