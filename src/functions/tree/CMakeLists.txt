add_library(FunctionsTree STATIC
    math_functions.cpp
    math_functions.h
    decision_tree_single.cpp
    decision_tree_single.h
    decision_tree_XGBoost.cpp
    decision_tree_XGBoost.h
    feature_importance.cpp
    feature_importance.h
    tree_visualization.cpp
    tree_visualization.h
)

target_include_directories(FunctionsTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(FunctionsTree PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
