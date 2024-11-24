# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(main_data_clean VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Créer la bibliothèque DataClean
add_library(DataClean
    data_clean.cpp
)

# Configurer les chemins d'inclusion et les dépendances
target_include_directories(DataClean PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/functions_tree
)

target_link_libraries(DataClean
    PUBLIC
    FunctionsTree
)

# Ajouter l'exécutable principal
add_executable(DataCleanMain main_data_clean.cpp)
target_link_libraries(DataCleanMain 
    PRIVATE 
    DataClean 
    FunctionsTree
)

# Configuration des tests
add_executable(DataCleanTests test_data_clean.cpp)
target_link_libraries(DataCleanTests
    PRIVATE
    GTest::GTest
    GTest::Main
    DataClean
    FunctionsTree
)

# Ajouter le test à CTest
add_test(NAME DataCleanTests COMMAND DataCleanTests)

