add_library(Boosting_LightGBM STATIC my_lightgbm.cpp)


target_include_directories(Boosting_LightGBM PUBLIC
    ${LGBM_INCLUDE}
)