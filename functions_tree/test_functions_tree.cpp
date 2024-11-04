#include "functions_tree.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <cassert>  // For assertion testing

using namespace std;

// Test for calculateVariance
TEST(VarianceTests, NormalValues) {
    std::vector<int> values1 = {1, 2, 3, 4, 5};
    float variance1 = calculateVariance(values1);
    EXPECT_FLOAT_EQ(variance1, 2.0);
}

TEST(VarianceTests, SameValues) {
    std::vector<int> values2 = {5, 5, 5, 5};
    float variance2 = calculateVariance(values2);
    EXPECT_FLOAT_EQ(variance2, 0.0);
}

TEST(VarianceTests, EmptyVector) {
    std::vector<int> empty_values = {};
    float variance_empty = calculateVariance(empty_values);
    EXPECT_FLOAT_EQ(variance_empty, 0.0); 
}

// Tests for getMaxFeature
TEST(GetMaxFeatureTests, NormalValues) {
    Matrix values = {
        {3, -1, 2},
        {1, 5, 0},
        {4, 0, -3}
    };
    EXPECT_EQ(getMaxFeature(values, 0), 4); // Max in first column
    EXPECT_EQ(getMaxFeature(values, 1), 5); // Max in second column
    EXPECT_EQ(getMaxFeature(values, 2), 2); // Max in third column
}

TEST(GetMaxFeatureTests, EmptyMatrix) {
    Matrix values = {};
    EXPECT_THROW(getMaxFeature(values, 0), std::out_of_range); // Should return 0 for empty matrix
}

TEST(GetMaxFeatureTests, FeatureIndexOutOfBounds) {
    Matrix values = {
        {1, 2, 3},
        {4, 5, 6}
    };
    // Here we check the behavior for out-of-bounds index
    EXPECT_THROW(getMaxFeature(values, 3), std::out_of_range); // Assuming you handle out-of-bounds correctly
    EXPECT_THROW(getMaxFeature(values, -1), std::out_of_range); // Negative index
}

// Tests for getMinFeature
TEST(GetMinFeatureTests, NormalValues) {
    Matrix values = {
        {3, -1, 2},
        {1, 5, 0},
        {4, 0, -3}
    };
    EXPECT_EQ(getMinFeature(values, 0), 1); // Min in first column
    EXPECT_EQ(getMinFeature(values, 1), -1); // Min in second column
    EXPECT_EQ(getMinFeature(values, 2), -3); // Min in third column
}

TEST(GetMinFeatureTests, EmptyMatrix) {
    Matrix values = {};
    EXPECT_THROW(getMinFeature(values, 0), std::out_of_range); // Should return 0 for empty matrix
}

TEST(GetMinFeatureTests, FeatureIndexOutOfBounds) {
    Matrix values = {
        {1, 2, 3},
        {4, 5, 6}
    };
    // Here we check the behavior for out-of-bounds index
    EXPECT_THROW(getMaxFeature(values, 3), std::out_of_range); // Assuming you handle out-of-bounds correctly
    EXPECT_THROW(getMaxFeature(values, -1), std::out_of_range); // Negative index
}

// Tests for getMeanFeature
TEST(GetMeanFeatureTests, NormalValues) {
    Matrix values = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    // Calculation of the average for the first characteristic (column 0)
    EXPECT_FLOAT_EQ(getMeanFeature(values, 0), 4.0f); // (1 + 4 + 7) / 3 = 4.0
    // Calculation of the average for the second characteristic (column 1)
    EXPECT_FLOAT_EQ(getMeanFeature(values, 1), 5.0f); // (2 + 5 + 8) / 3 = 5.0
    // Calculation of the average for the third characteristic (column 2)
    EXPECT_FLOAT_EQ(getMeanFeature(values, 2), 6.0f); // (3 + 6 + 9) / 3 = 6.0
}

TEST(GetMeanFeatureTests, EmptyMatrix) {
    Matrix values = {};
    EXPECT_THROW(getMinFeature(values, 0), std::out_of_range); // Should return 0 for empty matrix
}

TEST(GetMeanFeatureTests, FeatureIndexOutOfBounds) {
    Matrix values = {
        {1, 2, 3},
        {4, 5, 6}
    };
    // Here we check the behavior for out-of-bounds index
    EXPECT_THROW(getMaxFeature(values, 3), std::out_of_range); // Assuming you handle out-of-bounds correctly
    EXPECT_THROW(getMaxFeature(values, -1), std::out_of_range); // Negative index
}

// Tests for drawUniqueNumbers
TEST(DrawUniqueNumbersTests, HandlesMoreThanAvailableRows) {
    std::vector<int> result = drawUniqueNumbers(5, 3);
    EXPECT_EQ(result.size(), 3); // Should return all rows
}

TEST(DrawUniqueNumbersTests, HandlesExactRows) {
    int n = 3;
    int rows = 3;
    std::vector<int> result = drawUniqueNumbers(n, rows);

    // Check that we received exactly `n` elements
    EXPECT_EQ(result.size(), n);

    // Check that elements are within the expected range [0, rows - 1]
    for (int value : result) {
        EXPECT_GE(value, 0);
        EXPECT_LT(value, rows);
        
        // Ensure each element appears only once in the result
        EXPECT_EQ(std::count(result.begin(), result.end(), value), 1);
    }
}

// Tests for compareThresholds
TEST(CompareThresholdsTest, ReturnsMinVarianceThreshold) {
    // Create multiple thresholds with different weighted variances
    std::vector<Threshold> thresholds = {
        Threshold(0, 10, 5.0),  // Feature 0, threshold value 10, variance 5.0
        Threshold(1, 20, 2.5),  // Feature 1, threshold value 20, variance 2.5 (smallest)
        Threshold(2, 15, 7.0),  // Feature 2, threshold value 15, variance 7.0
        Threshold(0, 30, 4.0)   // Feature 0, threshold value 30, variance 4.0
    };

    // Call the compareThresholds function
    Threshold best_threshold = compareThresholds(thresholds);

    // Check that the returned threshold is indeed the one with the smallest weighted variance
    EXPECT_EQ(best_threshold.feature_index, 1);
    EXPECT_EQ(best_threshold.value, 20);
    EXPECT_FLOAT_EQ(best_threshold.weighted_variance, 2.5);
}

TEST(CompareThresholdsTest, HandlesSameVariancesThreshold) {
    // Create multiple thresholds with the same weighted variances
    std::vector<Threshold> thresholds = {
        Threshold(0, 10, 3.5),  // Feature 0, threshold value 10, variance 3.5
        Threshold(1, 20, 3.5),  // Feature 1, threshold value 20, variance 3.5
        Threshold(2, 15, 3.5),  // Feature 2, threshold value 15, variance 3.5
    };

    // Call the compareThresholds function
    Threshold best_threshold = compareThresholds(thresholds);

    // Check that the returned threshold is indeed the first one with the smallest weighted variance
    EXPECT_EQ(best_threshold.feature_index, 0);
    EXPECT_EQ(best_threshold.value, 10);
    EXPECT_FLOAT_EQ(best_threshold.weighted_variance, 3.5);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}