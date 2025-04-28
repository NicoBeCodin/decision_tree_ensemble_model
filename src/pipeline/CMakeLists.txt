add_library(Pipeline
    model_params.cpp
    run_models.cpp
    data_split.cpp
    getModelParameters.cpp
)

target_include_directories(Pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Pipeline
    FunctionsIO
    FunctionsTree
    FunctionsFeature
    FunctionsMath
    Bagging
    Boosting
    Boosting_LightGBM
    model_comparison
    LossFunction
)