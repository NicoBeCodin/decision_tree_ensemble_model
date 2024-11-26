# Version minimale de CMake
cmake_minimum_required(VERSION 3.10)

# Nom du projet
project(XGBoostCpp VERSION 1.0)

# Définir le standard C++ utilisé
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Inclure tous les fichiers sources et headers
set(SOURCES
    main.cpp
    xgboost.cpp
)

set(HEADERS
    xgboost.h
)

# Ajouter l'exécutable
add_executable(XGBoostCpp ${SOURCES} ${HEADERS})

# Option pour le mode Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(XGBoostCpp PRIVATE -g -O0)
endif()

# Afficher les messages à la compilation
message(STATUS "Configuration de projet : ${PROJECT_NAME}")
message(STATUS "Version de projet       : ${PROJECT_VERSION}")
message(STATUS "Mode de construction    : ${CMAKE_BUILD_TYPE}")
