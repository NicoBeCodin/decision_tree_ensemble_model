cmake_minimum_required(VERSION 3.10)

# Project name
project(Projet_PPN)

# Enables the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Added compilation options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Declaration of source files
set(SOURCES main.cpp functions.cpp)

# Creating the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Allows to use the "make clean" command
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM true)