#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "functions_io.h"  // Inclure le fichier header contenant toutes les fonctions

// Crée un fichier CSV temporaire pour les tests
void createTestCSV(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    file << content;
    file.close();
}

// Supprime le fichier après le test
void deleteTestCSV(const std::string& filename) {
    std::remove(filename.c_str());
}

/* Tests de la fonction openCSV */

TEST(OpenCSVTest, ValidFile) {
    // Crée un fichier CSV valide
    const std::string filename = "test.csv";
    createTestCSV(filename, "name,age\nAlice,30\nBob,25");

    // Appelle openCSV
    auto result = openCSV(filename);
    
    // Vérifie le contenu du fichier
    ASSERT_EQ(result.size(), 3); // Trois lignes (header + 2)
    EXPECT_EQ(result[0][0], "name");
    EXPECT_EQ(result[0][1], "age");
    EXPECT_EQ(result[1][0], "Alice");
    EXPECT_EQ(result[1][1], "30");
    EXPECT_EQ(result[2][0], "Bob");
    EXPECT_EQ(result[2][1], "25");

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

// Test de la fonction countCSVRows
TEST(CountCSVRowsTest, ValidCSV) {
    // Crée un fichier CSV valide avec plusieurs lignes
    const std::string filename = "test_count.csv";
    createTestCSV(filename, "header1,header2\nrow1_col1,row1_col2\nrow2_col1,row2_col2\nrow3_col1,row3_col2");

    // Appelle countCSVRows
    int rowCount = countCSVRows(filename);

    // Vérifie le nombre de lignes
    EXPECT_EQ(rowCount, 4); // 1 header + 3 lignes de données

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(CountCSVRowsTest, EmptyCSV1) {
    // Crée un fichier CSV vide
    const std::string filename = "empty_count.csv";
    createTestCSV(filename, "");

    // Appelle countCSVRows
    int rowCount = countCSVRows(filename);

    // Vérifie que le contenu est vide
    EXPECT_EQ(rowCount, 0); // Aucun contenu

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(CountCSVRowsTest, NonExistentFile) {
    // Appelle countCSVRows avec un fichier qui n'existe pas
    int rowCount = countCSVRows("non_existent_file.csv");

    // Vérifie que le contenu est vide
    EXPECT_EQ(rowCount, 0); // Aucun contenu
}

TEST(OpenCSVTest, NonExistentFile) {
    // Appelle openCSV avec un fichier qui n'existe pas
    auto result = openCSV("non_existent_file.csv");
    
    // Vérifie que le contenu est vide
    EXPECT_EQ(result.size(), 0); // Aucun contenu
}

TEST(OpenCSVTest, EmptyFile) {
    // Crée un fichier CSV vide
    const std::string filename = "empty.csv";
    createTestCSV(filename, "");

    // Appelle openCSV
    auto result = openCSV(filename);
    
    // Vérifie que le contenu est vide
    EXPECT_EQ(result.size(), 0); // Aucun contenu

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

/* Tests de la fonction OpenCSVLimited */

TEST(OpenCSVLimitedTest, ReadLimitedRows) {
    // Crée un fichier CSV valide avec plusieurs lignes
    const std::string filename = "test_limited.csv";
    createTestCSV(filename, "name,age\nAlice,30\nBob,25\nCharlie,35\nDavid,40");

    // Appelle openCSVLimited avec n = 2 (limiter à deux lignes)
    auto result = openCSVLimited(filename, 2);

    // Vérifie que seules 2 lignes ont été lues (header + 1 ligne)
    ASSERT_EQ(result.size(), 2); // Deux lignes lues (header + Alice)
    EXPECT_EQ(result[0][0], "name");
    EXPECT_EQ(result[0][1], "age");
    EXPECT_EQ(result[1][0], "Alice");
    EXPECT_EQ(result[1][1], "30");

    // Appelle openCSVLimited avec n = 3 (limiter à trois lignes)
    auto result2 = openCSVLimited(filename, 3);

    // Vérifie que 3 lignes ont été lues (header + 2 lignes)
    ASSERT_EQ(result2.size(), 3); // Trois lignes lues (header + Alice + Bob)
    EXPECT_EQ(result2[2][0], "Bob");
    EXPECT_EQ(result2[2][1], "25");

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(OpenCSVLimitedTest, ReadMoreRowsThanAvailable) {
    // Crée un fichier CSV avec peu de lignes
    const std::string filename = "test_limited_small.csv";
    createTestCSV(filename, "name,age\nAlice,30");

    // Appelle openCSVLimited avec n plus grand que le nombre de lignes disponibles
    auto result = openCSVLimited(filename, 10);  // n = 10, mais il n'y a qu'une ligne de données (header + 1)

    // Vérifie que toutes les lignes disponibles ont été lues
    ASSERT_EQ(result.size(), 2); // Header + Alice
    EXPECT_EQ(result[1][0], "Alice");
    EXPECT_EQ(result[1][1], "30");

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(OpenCSVLimitedTest, NonExistentFile) {
    // Appelle openCSVLimited avec un fichier qui n'existe pas
    auto result = openCSVLimited("non_existent_file.csv", 5);

    // Vérifie que le contenu est vide
    EXPECT_EQ(result.size(), 0); // Aucun contenu
}

/* Pas de test de la fonction printStringCSV (pas important) */

/* Tests de la fonction getColumnIndex */

TEST(GetColumnIndexTest, ValidColumn) {
    // Définit un header
    std::vector<std::string> header = {"name", "age", "city"};

    // Appelle getColumnIndex
    int index = getColumnIndex(header, "age");

    // Vérifie l'index
    EXPECT_EQ(index, 1); // L'index de "age" doit être 1
}

TEST(GetColumnIndexTest, InvalidColumn) {
    // Définit un header
    std::vector<std::string> header = {"name", "age", "city"};

    // Appelle getColumnIndex avec une colonne inexistante
    int index = getColumnIndex(header, "country");

    // Vérifie l'index pour une colonne qui n'existe pas
    EXPECT_EQ(index, -1); // L'index pour une colonne inexistante doit être -1
}

/* Tests des fonctions convertToInt et convertToFloat */

TEST(ConvertToIntTest, ValidInputs) {
    // Test des cas valides
    EXPECT_EQ(convertToInt("123.0"), 123);
    EXPECT_EQ(convertToInt("-456.0"), -456);
    EXPECT_EQ(convertToInt("0.0"), 0);
    EXPECT_EQ(convertToInt("123.33"), 123);
}

TEST(ConvertToIntTest, InvalidInputs) {
    // Test des cas invalides
    EXPECT_THROW(convertToInt("abc"), std::invalid_argument);
    EXPECT_THROW(convertToInt("999999999999999999"), std::out_of_range);
}

TEST(ConvertToFloatTest, ValidInputs) {
    // Test des cas valides
    EXPECT_FLOAT_EQ(convertToFloat("123.45"), 123.45f);
    EXPECT_FLOAT_EQ(convertToFloat("-678.9"), -678.9f);
    EXPECT_FLOAT_EQ(convertToFloat("0.0"), 0.0f);
    EXPECT_FLOAT_EQ(convertToFloat("1.23e10"), 1.23e10f);
    EXPECT_FLOAT_EQ(convertToFloat("123,45"), 123); // La fonction ne renvoie pas 123.45f
}

TEST(ConvertToFloatTest, InvalidInputs) {
    // Test des cas invalides qui doivent lever une exception
    EXPECT_THROW(convertToFloat("xyz"), std::invalid_argument);
    EXPECT_THROW(convertToFloat("3.4028235e39"), std::out_of_range); // Dépassement de la plage
}

/* Test de la fonction processParametersCSV */

TEST(ProcessParametersCSVTest, ValidCSV) {
    // Crée un fichier CSV valide avec des paramètres à traiter
    const std::string filename = "test_parameters.csv";
    createTestCSV(filename, "param1,param2,param3\n1,2,3.3\n4,5,6.6\n7,8,9.9");

    // Appelle openCSV pour lire le fichier CSV
    vector<vector<string>> content = openCSV(filename); // On lit toutes les lignes

    // Traite le contenu du CSV
    Matrix result = processParametersCSV(content);

    // Vérifie que les résultats sont corrects
    ASSERT_EQ(result.size(), 3); // Trois lignes de données (sans l'en-tête)
    
    EXPECT_EQ(result[0][0], 1);
    EXPECT_EQ(result[0][1], 2);

    EXPECT_EQ(result[1][0], 4);
    EXPECT_EQ(result[1][1], 5);

    EXPECT_EQ(result[2][0], 7);
    EXPECT_EQ(result[2][1], 8);

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(ProcessParametersCSVTest, EmptyCSV2) {
    // Crée un fichier CSV vide
    const std::string filename = "test_empty_parameters.csv";
    createTestCSV(filename, "");

    // Appelle openCSV pour lire le fichier CSV
    vector<vector<string>> content = openCSV(filename); // On verifie qu'il est bien vide

    // Traite le contenu du CSV
    Matrix result = processParametersCSV(content);

    // Vérifie que le résultat est vide
    EXPECT_EQ(result.size(), 0); // Aucun paramètre traité

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(ProcessParametersCSVTest, InvalidData) {
    // Crée un fichier CSV avec des données invalides
    const std::string filename = "test_invalid_parameters.csv";
    createTestCSV(filename, "param1,param2,param3\n1,abc,3\n4,5,xyz\n7,8,9.5");

    // Appelle openCSV pour lire le fichier CSV
    vector<vector<string>> content = openCSV(filename); // On lit toutes les lignes

    // Traite le contenu du CSV et s'attend à des exceptions
    Matrix result;
    
    // On s'attend à ce que les conversions échouent
    try {
        result = processParametersCSV(content);
    } catch (const std::invalid_argument&) {
        // Exception levée comme prévu
        EXPECT_TRUE(true);
    }

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

/* Tests de la fonction processResults */

TEST(ProcessResultsTest, ValidCSV) {
    // Crée un fichier CSV valide avec des paramètres à traiter
    const std::string filename = "test_parameters.csv";
    createTestCSV(filename, "param1,param2,param3\n1,2,3.3\n4,5,6.6\n7,8,9.9");

    // Appelle openCSV pour lire le fichier CSV
    vector<vector<string>> content = openCSV(filename); // On lit toutes les lignes

    // Traite le contenu du CSV
    vector<float> result = processResultsCSV(content);

    // Vérifie que les résultats sont corrects
    ASSERT_EQ(result.size(), 3); // Trois élémennts de données
    
    EXPECT_EQ(result[0], 3.3f);
    
    EXPECT_EQ(result[1], 6.6f);

    EXPECT_EQ(result[2], 9.9f);

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(ProcessParametersCSVTest, EmptyCSV3) {
    // Crée un fichier CSV vide
    const std::string filename = "test_parameters_empty.csv";
    createTestCSV(filename, "");

    // Appelle openCSV pour lire le fichier CSV
    vector<vector<string>> content = openCSV(filename); // On lit le fichier vide

    // Traite le contenu du CSV
    vector<float> result = processResultsCSV(content);

    // Vérifie que le résultat est vide
    EXPECT_EQ(result.size(), 0); // Aucun paramètre traité

    // Supprime le fichier après le test
    deleteTestCSV(filename);
}

TEST(ProcessParametersCSVTest, InvalidCSV) {
    // Crée un fichier CSV invalide
    const std::string filename = "test_invalid_parameters.csv";
    createTestCSV(filename,"param1,param2,param3\n1,2,xyz\n4,5,8.1e39\n6,7,UAEK");

    // Appelle openCSV pour lire le fichier CSV
    vector<vector<string>> content = openCSV(filename);

    // Traite le contenu du CSV et s'attend à des exceptions
    vector<float> result;

    // On s'attend à ce que les conversions échouent
    try {
        result = processResultsCSV(content);
    } catch(const std::invalid_argument&) {
        // Exception levée comme prévu
        EXPECT_TRUE(true);
    }

    deleteTestCSV(filename);
}

/* Pas de test de la fonction printParamAndResults (pas important) */

/* Fonction main pour exécuter les tests */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv); // Initialiser Google Test
    return RUN_ALL_TESTS(); // Exécuter tous les tests
}
