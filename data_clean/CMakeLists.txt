# Version minimale de CMake
cmake_minimum_required(VERSION 3.10)

# Définir le standard C++ utilisé
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Inclure tous les fichiers sources et headers
set(SOURCES
    main_data_clean.cpp
    data_clean.cpp
    ../functions_tree/math_functions.cpp
)

set(HEADERS
    data_clean.h
    ../functions_tree/math_functions.h
)
# Ajouter l'exécutable
add_executable(DATACLEAN ${SOURCES} ${HEADERS})