add_library(FunctionsFeature STATIC
    feature_importance.cpp
    feature_importance.h
)

target_include_directories(FunctionsFeature PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(FunctionsFeature PUBLIC FunctionsMath)