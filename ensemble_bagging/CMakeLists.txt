add_library(Bagging bagging.cpp)

# Include header file path
target_include_directories(Bagging PUBLIC ${CMAKE_SOURCE_DIR}/functions_tree)

# Link dependencies
target_link_libraries(Bagging PUBLIC FunctionsTree)
