cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src/functions/io)
add_subdirectory(src/functions/tree)
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
    )

    add_executable(decision_tree_test tests/decision_tree_test.cpp)
    target_link_libraries(decision_tree_test
        gtest_main
        FunctionsTree
    )

    add_executable(boosting_test tests/boosting_test.cpp)
    target_link_libraries(boosting_test
        gtest_main
        FunctionsTree
        Boosting
    )

    add_executable(xgboost_test tests/xgboost_test.cpp)
    target_link_libraries(xgboost_test
        gtest_main
        FunctionsTree
        Boosting
        Boosting_XGBoost
    )

    add_executable(bagging_test tests/bagging_test.cpp)
    target_link_libraries(bagging_test
        gtest_main
        FunctionsTree
        Bagging
        Boosting
    )

    add_executable(cross_validation_test tests/cross_validation_test.cpp)
    target_link_libraries(cross_validation_test
        gtest_main
        FunctionsTree
        Bagging
        Boosting
        Boosting_XGBoost
    )
endif()

add_executable(MainDataClean src/main/main_data_clean.cpp)
add_executable(MainEnsemble src/main/main.cpp)
add_executable(MainKFold src/main/main_kfold.cpp)
add_executable(MainComparison src/main/main_comparison.cpp)

target_link_libraries(MainDataClean PRIVATE Data_Clean)

target_link_libraries(MainEnsemble PRIVATE
    FunctionsIO
    FunctionsTree
    Bagging
    Boosting
    Boosting_XGBoost
    model_comparison
    Data_Clean
)

target_link_libraries(MainKFold PRIVATE
    FunctionsIO
    FunctionsTree
    Bagging
    Boosting
)

target_include_directories(MainDataClean PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
)

target_include_directories(MainEnsemble PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting_XGBoost
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
)

target_include_directories(MainKFold PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
)

target_include_directories(MainComparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/tree
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting_XGBoost
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
)