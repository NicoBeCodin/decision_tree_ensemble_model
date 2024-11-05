
add_library(Bagging bagging.cpp)


target_include_directories(Bagging PUBLIC ${CMAKE_SOURCE_DIR}/functions_tree)

target_link_libraries(Bagging PUBLIC RegressionTree)
