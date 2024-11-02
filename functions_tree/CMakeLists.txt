# Compile regression_tree.cpp as a static library named RegressionTree
add_library(RegressionTree regression_tree.cpp)

# Include header file path
target_include_directories(RegressionTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
