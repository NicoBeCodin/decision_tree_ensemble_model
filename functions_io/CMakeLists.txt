# Compile functions_io.cpp as a static library named FunctionsIO
add_library(FunctionsIO functions_io.cpp)

# Include header file path
target_include_directories(FunctionsIO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add test executable
add_executable(FunctionsIOTests test_functions_io.cpp)
target_link_libraries(FunctionsIOTests
    PRIVATE
    GTest::GTest
    GTest::Main
    FunctionsIO
)

# Add test to CTest
add_test(NAME FunctionsIOTests COMMAND FunctionsIOTests)
