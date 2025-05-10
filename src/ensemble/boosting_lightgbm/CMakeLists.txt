
add_library(Boosting_LightGBM STATIC my_lightgbm.cpp)


target_include_directories(Boosting_LightGBM PUBLIC
    ${LGBM_INCLUDE_DIR}
)

target_link_libraries(Boosting_LightGBM PUBLIC
    ${LGBM_LIB}
    ${OPENMP_LIB}
)
