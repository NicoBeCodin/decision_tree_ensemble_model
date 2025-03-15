add_library(Boosting_XGBoost
    boosting_XGBoost.cpp
)
# Lier OpenMP seulement si activé
if(USE_OPENMP)
    target_link_libraries(Boosting_XGBoost PRIVATE ${OPENMP_LIB})
endif()

target_include_directories(Boosting_XGBoost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Boosting_XGBoost PUBLIC FunctionsTree)
target_link_libraries(Boosting_XGBoost PUBLIC FunctionsMath)