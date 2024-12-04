add_library(FunctionsTree
    math_functions.cpp
    decision_tree_single.cpp
    decision_tree_XGBoost.cpp
)

target_include_directories(FunctionsTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
