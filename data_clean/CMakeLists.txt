# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(main_data_clean VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Add the library object file (data_clean.o equivalent)
add_library(data_clean OBJECT data_clean.cpp)

# Add the executable and link it with the object library
add_executable(main_data_clean main_data_clean.cpp $<TARGET_OBJECTS:data_clean>)

