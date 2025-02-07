cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src/functions/io)
add_subdirectory(src/functions/tree)
add_subdirectory(src/functions/loss)
add_subdirectory(src/functions/math)
add_subdirectory(src/functions/feature)
add_subdirectory(src/ensemble/bagging)
add_subdirectory(src/ensemble/boosting)
add_subdirectory(src/ensemble/boosting_XGBoost)
add_subdirectory(src/data_clean)
add_subdirectory(src/model_comparison)


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

    add_executable(xgboost_test tests/xgboost_test.cpp)
    target_link_libraries(xgboost_test
        gtest_main
        FunctionsTree
        FunctionsFeature
        FunctionsMath
        Boosting
        Boosting_XGBoost
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
        Boosting_XGBoost
        LossFunction
    )
endif()

add_executable(MainDataClean src/main/main_data_clean.cpp)
add_executable(MainEnsemble src/main/main.cpp src/main/utility.cpp)
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
    Boosting_XGBoost
    model_comparison
    Data_Clean
    LossFunction
)

target_link_libraries(MainKFold PRIVATE
    FunctionsIO
    FunctionsTree
    FunctionsFeature
    FunctionsMath
    Bagging
    Boosting
    LossFunction
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting_XGBoost
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main
    
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting_XGBoost
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
)