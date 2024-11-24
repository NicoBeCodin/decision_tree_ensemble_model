add_library(Boosting
    boosting.cpp
    loss_function.cpp
)

# Include header file path
target_include_directories(Boosting PUBLIC 
    ${CMAKE_SOURCE_DIR}/functions_tree
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(Boosting PUBLIC FunctionsTree)

# Add test executable
add_executable(BoostingTests test_boosting.cpp)
target_link_libraries(BoostingTests
    PRIVATE
    GTest::GTest
    GTest::Main
    Boosting
)

# Add test to CTest
add_test(NAME BoostingTests COMMAND BoostingTests)
