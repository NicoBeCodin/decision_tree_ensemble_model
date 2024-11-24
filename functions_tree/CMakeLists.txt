add_library(FunctionsTree
    math_functions.cpp
    regression_tree.cpp
    decision_tree_single.cpp
)

target_include_directories(FunctionsTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Remplacer les trois exécutables de test par un seul
add_executable(TreeTests test_tree.cpp)

# Lier le test avec les bibliothèques nécessaires
target_link_libraries(TreeTests
    PRIVATE
    GTest::GTest
    GTest::Main
    FunctionsTree
)

# Ajouter le test à CTest
add_test(NAME TreeTests COMMAND TreeTests)
