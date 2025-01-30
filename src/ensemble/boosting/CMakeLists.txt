add_library(Boosting
    boosting.cpp
    loss_function.cpp
)

target_include_directories(Boosting PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Boosting PUBLIC FunctionsTree)
