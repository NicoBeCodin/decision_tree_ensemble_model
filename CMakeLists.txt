cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(functions_io)
add_subdirectory(functions_tree)
add_subdirectory(ensemble_bagging)
add_subdirectory(ensemble_boosting)
add_subdirectory(ensemble_boosting_XGBoost)
add_subdirectory(data_clean)
add_subdirectory(model_comparison)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(DataClean main_data_clean.cpp)
add_executable(MainEnsemble main.cpp)
add_executable(MainKFold main_kfold.cpp)

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
)

add_executable(cross_validation_test tests/cross_validation_test.cpp)
target_link_libraries(cross_validation_test
    gtest_main
    FunctionsTree
    Bagging
    Boosting
    Boosting_XGBoost
)

target_link_libraries(DataClean PRIVATE Data_Clean)

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

target_include_directories(MainEnsemble PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_io
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_tree
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_boosting_XGBoost
    ${CMAKE_CURRENT_SOURCE_DIR}/model_comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/data_clean
)

target_include_directories(MainKFold PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_io
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_tree
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_boosting
)

# Main comparison executable
add_executable(decision_tree main_comparison.cpp)

target_link_libraries(decision_tree PRIVATE
    model_comparison
    FunctionsTree
    Bagging
    Boosting
    Boosting_XGBoost
)

target_include_directories(decision_tree PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
