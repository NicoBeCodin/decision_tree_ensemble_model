cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(functions_io)
add_subdirectory(functions_tree)
add_subdirectory(ensemble_bagging)
add_subdirectory(ensemble_boosting)

add_executable(MainEnsemble main.cpp)
add_executable(MainKFold main_kfold.cpp)

target_link_libraries(MainEnsemble 
    FunctionsIO
    FunctionsTree
    Bagging
    Boosting
)

target_link_libraries(MainKFold 
    FunctionsIO
    FunctionsTree
    Bagging
    Boosting
)

target_include_directories(MainEnsemble PRIVATE
    functions_io
    functions_tree
    ensemble_bagging
    ensemble_boosting
)

target_include_directories(MainKFold PRIVATE
    functions_io
    functions_tree
    ensemble_bagging
    ensemble_boosting
)
