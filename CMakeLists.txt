# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add subdirectories for modules
add_subdirectory(functions_io)
add_subdirectory(functions_tree)
add_subdirectory(ensemble_bagging)
add_subdirectory(ensemble_boosting)

# Main executable
add_executable(MainEnsemble main.cpp)

# Link libraries to the executable
target_link_libraries(MainEnsemble PUBLIC FunctionsIO RegressionTree Bagging Boosting)

# Include directories
target_include_directories(MainEnsemble PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_io
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_tree
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_boosting
)
