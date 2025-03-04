#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "../src/data_clean/data_clean.h"

// Helper function to create a temporary CSV file
void createTestCSV(const std::string& filePath, const std::string& header, const std::vector<std::vector<double>>& data) {
    std::ofstream file(filePath);
    file << header << "\n";
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

// Test fixture class
class DataCleanTest : public ::testing::Test {
protected:
    std::string filePath = "test.csv";
    std::string header = "p5,p6,matrix_size_x,matrix_size_y,p2,p1,p8,p7,p3,p4,performance";
    std::vector<std::vector<double>> data = {
        {18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217},
        {24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895},
        {30, 28, 1619, 1857, 163, 102, 71, 71, 16, 25, 0.0272806}
    };

    void SetUp() override {
        createTestCSV(filePath, header, data);
    }
};

TEST_F(DataCleanTest, ReadCSVTest) {
    std::string readHeader;
    int rowLength;
    std::vector<double> dataset = readCSV(filePath, readHeader, rowLength);

    EXPECT_EQ(readHeader, header);
    EXPECT_EQ(rowLength, 11);
    EXPECT_EQ(dataset.size(), 33); // 3 rows * 11 columns
}

TEST_F(DataCleanTest, RemoveOutliersTest) {
    std::vector<double> dataset = {
        18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217,
        24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895,
        30, 28, 1619, 1857, 163, 102, 71, 71, 16, 25, 0.0272806
    };
    int rowLength = 11;
    double threshold = 2.0;

    std::vector<double> cleanedData = removeOutliers(dataset, rowLength, threshold);

    EXPECT_EQ(cleanedData.size(), dataset.size());
}

TEST_F(DataCleanTest, RemoveOutliersByBinningTest) {
    std::vector<double> dataset = {
        18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217,
        24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895,
        30, 28, 1619, 1857, 163, 102, 71, 71, 16, 25, 0.0272806
    };
    int rowLength = 11;
    int numBins = 2;
    double zThreshold = 2.0;

    std::vector<double> cleanedData = removeOutliersByBinning(dataset, rowLength, numBins, zThreshold);
    EXPECT_LE(cleanedData.size(), dataset.size());
}

TEST_F(DataCleanTest, WriteCSVTest) {
    std::string outputFilePath = "output.csv";
    std::vector<double> dataset = {
        18, 8, 2539, 3969, 139, 144, 92, 40, 17, 18, 0.0388217,
        24, 16, 4005, 3526, 84, 240, 47, 60, 27, 6, 0.0646895
    };
    int rowLength = 11;

    writeCSV(outputFilePath, header, dataset, rowLength);

    std::string readHeader;
    int readRowLength;
    std::vector<double> readDataset = readCSV(outputFilePath, readHeader, readRowLength);

    EXPECT_EQ(readHeader, header);
    EXPECT_EQ(readRowLength, rowLength);
    EXPECT_EQ(readDataset, dataset);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}