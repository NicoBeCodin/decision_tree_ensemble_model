cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

# Add this very early in the CMakeLists.txt
option(ENABLE_LIGHTGBM "Enable LightGBM integration" ON)

if (ENABLE_LIGHTGBM)
    if (NOT LGBM_LIB)
        find_library(LGBM_LIB NAMES lightgbm lightgbm _lightgbm lib_lightgbm PATHS
            /usr/local/lib
            /usr/lib
            "$ENV{HOME}/lightgbm_install/lib"
            NO_DEFAULT_PATH
        )
    endif()

    if (NOT LGBM_LIB)
        message(FATAL_ERROR "LightGBM library (lib_lightgbm.so) not found. Please run 'sudo make install' for LightGBM, or specify the path manually with -DLGBM_LIB=...")
    else()
        message(STATUS "Found LightGBM library: ${LGBM_LIB}")
    endif()

    if (EXISTS "/usr/local/include/LightGBM")
        set(LGBM_INCLUDE "/usr/local/include/LightGBM")
    else()
        message(FATAL_ERROR "LightGBM header directory not found (/usr/local/include/LightGBM).")
    endif()
endif()

# OpenMP setup
option(OPENMP "Enable OpenMP" ON)
if (OPENMP)
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        message(STATUS "OpenMP enabled")
        set(OPENMP_LIB OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP requested but not found")
        set(OPENMP_LIB "")
    endif()
else()
    message(STATUS "OpenMP disabled")
    set(OPENMP_LIB "")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

add_subdirectory(src/functions/io)
add_subdirectory(src/functions/tree)
add_subdirectory(src/functions/loss)
add_subdirectory(src/functions/math)
add_subdirectory(src/functions/feature)
add_subdirectory(src/ensemble/bagging)
add_subdirectory(src/ensemble/boosting)
if (ENABLE_LIGHTGBM)
  add_subdirectory(src/ensemble/boosting_lightgbm)
endif()
add_subdirectory(src/data_clean)
add_subdirectory(src/model_comparison)
add_subdirectory(src/pipeline)
add_subdirectory(src/utils)

# Option to enable/disable tests
option(ENABLE_TESTS "Enable building tests" ON)

if (ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Tests
    add_executable(math_functions_test tests/math_functions_test.cpp)
    target_link_libraries(math_functions_test
        gtest_main
        FunctionsTree
        FunctionsFeature
        FunctionsMath
    )

    add_executable(decision_tree_test tests/decision_tree_test.cpp)
    target_link_libraries(decision_tree_test
        gtest_main
        FunctionsTree
        FunctionsFeature
        FunctionsMath
    )

    add_executable(boosting_test tests/boosting_test.cpp)
    target_link_libraries(boosting_test
        gtest_main
        FunctionsTree
        FunctionsFeature
        FunctionsMath
        Boosting
        LossFunction
    )

    add_executable(bagging_test tests/bagging_test.cpp)
    target_link_libraries(bagging_test
        gtest_main
        FunctionsTree
        FunctionsFeature
        FunctionsMath
        Bagging
        Boosting
        LossFunction
    )

    add_executable(cross_validation_test tests/cross_validation_test.cpp)
    target_link_libraries(cross_validation_test
        gtest_main
        FunctionsTree
        FunctionsFeature
        FunctionsMath
        Bagging
        Boosting
        LossFunction
    )

    add_executable(data_clean_test tests/data_clean_test.cpp)
    target_link_libraries(data_clean_test
        gtest_main
        FunctionsMath
        FunctionsTree
        Data_Clean
    )
endif()

add_executable(MainDataClean src/main/main_data_clean.cpp)
add_executable(MainEnsemble src/main/main.cpp src/utils/utility.cpp)
add_executable(MainKFold src/main/main_kfold.cpp)
add_executable(MainComparison src/main/main_comparison.cpp)

target_link_libraries(MainDataClean PRIVATE Data_Clean)

target_link_libraries(MainEnsemble PRIVATE
    FunctionsIO
    FunctionsTree
    FunctionsFeature
    FunctionsMath
    Bagging
    Boosting
    model_comparison
    Data_Clean
    LossFunction
    Pipeline
    Utils
    ${OPENMP_LIB}
)

if (ENABLE_LIGHTGBM)
    target_link_libraries(MainEnsemble PRIVATE Boosting_LightGBM ${LGBM_LIB})
endif()

target_link_libraries(MainKFold PRIVATE
    FunctionsIO
    FunctionsTree
    FunctionsFeature
    FunctionsMath
    Bagging
    Boosting
    LossFunction
)

target_link_libraries(MainComparison PRIVATE
    Pipeline
)

target_include_directories(MainDataClean PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/feature
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
)

target_include_directories(MainEnsemble PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/feature
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/loss
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

target_include_directories(MainKFold PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/feature
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/loss
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting

)

target_include_directories(MainComparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/feature
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/loss
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
)