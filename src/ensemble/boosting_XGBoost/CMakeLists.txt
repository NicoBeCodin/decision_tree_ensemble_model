add_library(Boosting_XGBoost
    boosting_XGBoost.cpp
)
# ✅ Link OpenMP in this library
target_link_libraries(Boosting_XGBoost PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(Boosting_XGBoost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Boosting_XGBoost PUBLIC FunctionsTree)
target_link_libraries(Boosting_XGBoost PUBLIC FunctionsMath)