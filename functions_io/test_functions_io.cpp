#include "functions_io.h"
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// ====== Utility Functions ======
void createTestCSV(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    file << content;
    file.close();
}

void deleteTestCSV(const std::string& filename) {
    std::remove(filename.c_str());
}

// ====== DataIO Tests ======
TEST(DataIOTest, ValidFile) {
    const std::string filename = "test.csv";
    // Create CSV file with numeric values only
    createTestCSV(filename, "p1,p2,performance\n30.0,25.0,0.8\n35.0,28.0,0.9");

    DataIO dataIO;
    auto result = dataIO.readCSV(filename);
    
    ASSERT_EQ(result.first.size(), 2);  // Two rows of data
    ASSERT_EQ(result.first[0].size(), 2);  // Two features per row
    
    EXPECT_DOUBLE_EQ(result.first[0][0], 30.0);
    EXPECT_DOUBLE_EQ(result.first[0][1], 25.0);
    EXPECT_DOUBLE_EQ(result.second[0], 0.8);
    
    EXPECT_DOUBLE_EQ(result.first[1][0], 35.0);
    EXPECT_DOUBLE_EQ(result.first[1][1], 28.0);
    EXPECT_DOUBLE_EQ(result.second[1], 0.9);

    deleteTestCSV(filename);
}

// ====== Non Existent File Tests ======
TEST(DataIOTest, NonExistentFile) {
    DataIO dataIO;
    auto result = dataIO.readCSV("non_existent_file.csv");
    
    // Check if vectors are empty
    EXPECT_TRUE(result.first.empty());
    EXPECT_TRUE(result.second.empty());
}

// ====== Empty File Tests ======
TEST(DataIOTest, EmptyFile) {
    const std::string filename = "empty.csv";
    createTestCSV(filename, "");

    DataIO dataIO;
    auto result = dataIO.readCSV(filename);
    
    // Check if vectors are empty
    EXPECT_TRUE(result.first.empty());
    EXPECT_TRUE(result.second.empty());

    deleteTestCSV(filename);
}

// ====== Write Results Tests ======
TEST(DataIOTest, WriteResults) {
    const std::string filename = "results.csv";
    std::vector<double> results = {0.8, 0.9, 0.7};
    
    DataIO dataIO;
    dataIO.writeResults(results, filename);
    
    // Check if file was created and contains correct values
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());
    
    std::vector<double> readResults;
    std::string line;
    while (std::getline(file, line)) {
        readResults.push_back(std::stod(line));
    }
    
    ASSERT_EQ(readResults.size(), 3);
    EXPECT_DOUBLE_EQ(readResults[0], 0.8);
    EXPECT_DOUBLE_EQ(readResults[1], 0.9);
    EXPECT_DOUBLE_EQ(readResults[2], 0.7);
    
    file.close();
    deleteTestCSV(filename);
}
