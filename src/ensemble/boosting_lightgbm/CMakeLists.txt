# 编译 my_lightgbm.cpp 为静态库 Boosting_LightGBM
add_library(Boosting_LightGBM STATIC my_lightgbm.cpp)

# 公开 LightGBM 头文件
target_include_directories(Boosting_LightGBM PUBLIC
    ${LGBM_INCLUDE}
)

# 链接 LightGBM 动态库 + OpenMP（如果有）
target_link_libraries(Boosting_LightGBM PUBLIC
    ${LGBM_LIB}
    ${OPENMP_LIB}
)
