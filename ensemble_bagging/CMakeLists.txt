add_library(Bagging bagging.cpp)

# Include header file path
target_include_directories(Bagging PUBLIC ${CMAKE_SOURCE_DIR}/functions_tree ${CMAKE_CURRENT_SOURCE_DIR})

# Link dependencies
target_link_libraries(Bagging PUBLIC FunctionsTree)

# Add test executable
add_executable(BaggingTests test_bagging.cpp)
target_link_libraries(BaggingTests
    PRIVATE
    GTest::GTest
    GTest::Main
    Bagging
)

# Add test to CTest
add_test(NAME BaggingTests COMMAND BaggingTests)
