# Compile functions_io.cpp as a static library named FunctionsIO
add_library(FunctionsIO functions_io.cpp)

# Include header file path
target_include_directories(FunctionsIO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
