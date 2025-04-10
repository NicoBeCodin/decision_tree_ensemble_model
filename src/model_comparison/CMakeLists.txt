add_library(model_comparison
    model_comparison.cpp
)


target_include_directories(model_comparison PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(model_comparison
    FunctionsTree
    Bagging
    Boosting
) 