add_library(FunctionsTree
    regression_tree.cpp
    decision_tree_single.cpp
)

target_include_directories(FunctionsTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
