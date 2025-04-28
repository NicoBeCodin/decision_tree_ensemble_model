add_library(FunctionsMath STATIC
    math_functions.cpp
    math_functions.h
)

target_include_directories(FunctionsMath PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})