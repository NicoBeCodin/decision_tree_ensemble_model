# CMakeLists.txt in ensemble_boosting directory
add_library(Boosting
    boosting.cpp
    loss_function.cpp
)

# Include header file path
target_include_directories(Boosting PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/functions_tree)

# Link dependencies
target_link_libraries(Boosting PUBLIC RegressionTree)


