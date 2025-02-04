add_library(Boosting
    boosting.cpp
)

target_include_directories(Boosting PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Boosting PUBLIC FunctionsTree)
target_link_libraries(Boosting PUBLIC FunctionsMath)
target_link_libraries(Boosting PRIVATE LossFunction)