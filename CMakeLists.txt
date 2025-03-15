cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

# Option pour activer/désactiver OpenMP (par défaut ON)
option(USE_OPENMP "Activer le support OpenMP" ON)

# Détection du système
if(APPLE)
    message(STATUS "🍎 Système macOS détecté")
    set(IS_MACOS TRUE)
else()
    set(IS_MACOS FALSE)
endif()

# Configuration OpenMP
if(USE_OPENMP)
    if(IS_MACOS)
        # Configuration spécifique pour macOS
        message(STATUS "🔄 Configuration OpenMP pour macOS...")
        
        # Chercher libomp installé via Homebrew
        execute_process(
            COMMAND brew --prefix libomp
            RESULT_VARIABLE BREW_LIBOMP
            OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(BREW_LIBOMP EQUAL 0 AND EXISTS "${BREW_LIBOMP_PREFIX}")
            message(STATUS "✅ libomp trouvé à ${BREW_LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${BREW_LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${BREW_LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_C_FOUND TRUE)
            set(OpenMP_CXX_FOUND TRUE)
            
            # Définir la variable pour la liaison
            set(OPENMP_LIB "${OpenMP_omp_LIBRARY}")
            
            # Ajouter les flags de compilation
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            
            # Définir la macro pour le préprocesseur
            add_definitions(-DUSE_OPENMP)
            
            message(STATUS "✅ OpenMP activé pour macOS")
        else()
            message(WARNING "⚠️ libomp non trouvé. Installez-le avec 'brew install libomp'")
            message(STATUS "🚫 Désactivation d'OpenMP")
            set(USE_OPENMP OFF)
        endif()
    else()
        # Configuration standard pour Linux/Windows
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            # Définir la variable pour la liaison
            set(OPENMP_LIB OpenMP::OpenMP_CXX)
            
            # Ajouter les flags de compilation
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            
            # Définir la macro pour le préprocesseur
            add_definitions(-DUSE_OPENMP)
            
            message(STATUS "✅ OpenMP activé")
        else()
            message(WARNING "⚠️ OpenMP demandé mais non trouvé")
            message(STATUS "🚫 Désactivation d'OpenMP")
            set(USE_OPENMP OFF)
        endif()
    endif()
else()
    message(STATUS "🚫 OpenMP désactivé par l'utilisateur")
    # Définir une variable vide pour la liaison
    set(OPENMP_LIB "")
endif()

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

    add_executable(data_clean_test tests/data_clean_test.cpp)
    target_link_libraries(data_clean_test
        gtest_main
        FunctionsMath
        FunctionsTree
        Data_Clean
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
    ${OPENMP_LIB}
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