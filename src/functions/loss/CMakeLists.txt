add_library(LossFunction loss_function.cpp)

# Include header file path
target_include_directories(LossFunction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})