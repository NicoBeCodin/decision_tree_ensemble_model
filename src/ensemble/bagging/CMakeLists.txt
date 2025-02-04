add_library(Bagging bagging.cpp)

# Include header file path
target_include_directories(Bagging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link dependencies
target_link_libraries(Bagging PUBLIC FunctionsTree)
target_link_libraries(Bagging PRIVATE Boosting)
