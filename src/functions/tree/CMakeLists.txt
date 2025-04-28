add_library(FunctionsTree STATIC
    decision_tree_single.cpp
    decision_tree_single.h
    vizualization/tree_visualization.cpp
    vizualization/tree_visualization.h

)
target_include_directories(FunctionsTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(FunctionsTree PRIVATE OpenMP::OpenMP_CXX)


set_target_properties(FunctionsTree PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
