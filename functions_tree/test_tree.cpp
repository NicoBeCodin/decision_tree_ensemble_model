#include <gtest/gtest.h>
#include "regression_tree.h"
#include "math_functions.h"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>
#include <random>

// Define Matrix type
using Matrix = std::vector<std::vector<double>>;

// Define Threshold structure
struct Threshold {
    int feature_index;
    double value;
    double weighted_variance;
    
    Threshold(int idx, double val, double var) 
        : feature_index(idx), value(val), weighted_variance(var) {}
};


double getMaxFeature(const Matrix& values, int feature_index) {
    if (values.empty()) {
        throw std::out_of_range("Matrix is empty");
    }
    if (feature_index < 0 || feature_index >= static_cast<int>(values[0].size())) {
        throw std::out_of_range("Feature index out of bounds");
    }
    
    double max_val = values[0][feature_index];
    for (const auto& row : values) {
        max_val = std::max(max_val, row[feature_index]);
    }
    return max_val;
}

double getMinFeature(const Matrix& values, int feature_index) {
    if (values.empty()) {
        throw std::out_of_range("Matrix is empty");
    }
    if (feature_index < 0 || feature_index >= static_cast<int>(values[0].size())) {
        throw std::out_of_range("Feature index out of bounds");
    }
    
    double min_val = values[0][feature_index];
    for (const auto& row : values) {
        min_val = std::min(min_val, row[feature_index]);
    }
    return min_val;
}

double getMeanFeature(const Matrix& values, int feature_index) {
    if (values.empty()) {
        throw std::out_of_range("Matrix is empty");
    }
    if (feature_index < 0 || feature_index >= static_cast<int>(values[0].size())) {
        throw std::out_of_range("Feature index out of bounds");
    }
    
    double sum = 0.0;
    for (const auto& row : values) {
        sum += row[feature_index];
    }
    return sum / values.size();
}

std::vector<int> drawUniqueNumbers(int n, int rows) {
    std::vector<int> numbers(rows);
    std::iota(numbers.begin(), numbers.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (n >= rows) {
        return numbers;
    }
    
    std::vector<int> result;
    std::sample(numbers.begin(), numbers.end(), std::back_inserter(result), n, gen);
    return result;
}

Threshold compareThresholds(const std::vector<Threshold>& thresholds) {
    if (thresholds.empty()) {
        throw std::runtime_error("Empty thresholds vector");
    }
    
    return *std::min_element(thresholds.begin(), thresholds.end(),
        [](const Threshold& a, const Threshold& b) {
            return a.weighted_variance < b.weighted_variance;
        });
}

// ====== RegressionTree Tests ======
class RegressionTreeTest : public ::testing::Test {
protected:
    MeanSquaredError mse;
    RegressionTree* tree;

    void SetUp() override {
        tree = new RegressionTree(3, &mse);
    }

    void TearDown() override {
        delete tree;
    }
};

TEST_F(RegressionTreeTest, LinearDataTest) {
    std::vector<std::vector<double>> data = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    };
    std::vector<double> labels = {2.0, 4.0, 6.0, 8.0, 10.0};

    tree->train(data, labels);
    
    std::vector<double> test_sample = {3.5};
    double prediction = tree->predict(test_sample);

    EXPECT_GE(prediction, 6.0);
    EXPECT_LE(prediction, 8.0);
}

TEST_F(RegressionTreeTest, MultiFeatureTest) {
    std::vector<std::vector<double>> data = {
        {1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}, {2.0, 2.0}
    };
    std::vector<double> labels = {2.0, 3.0, 3.0, 4.0};

    tree->train(data, labels);
    
    std::vector<double> test_sample = {1.5, 1.5};
    double prediction = tree->predict(test_sample);

    EXPECT_GE(prediction, 2.0);
    EXPECT_LE(prediction, 4.0);
}

// ====== Mathematical Functions Tests ======
TEST(MathFunctionsTest, CalculateVarianceTest) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    double mean = Math::calculateMean(values);
    double variance = Math::calculateMSE(values);
    EXPECT_NEAR(variance, 2.0, 0.1);
}

TEST(MathFunctionsTest, EmptyInputTest) {
    std::vector<double> empty_values;
    double mean = Math::calculateMean(empty_values);
    double mse = Math::calculateMSE(empty_values);
    EXPECT_DOUBLE_EQ(mean, 0.0);
    EXPECT_DOUBLE_EQ(mse, 0.0);
}

TEST(MathFunctionsTest, CalculateStdDevTest) {
    std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double mean = Math::calculateMean(values);
    double stddev = Math::calculateStdDev(values, mean);
    EXPECT_NEAR(stddev, 2.0, 0.1);
}

TEST(MathFunctionsTest, NegativeGradientTest) {
    std::vector<double> y_true = {1.0, 2.0, 3.0};
    std::vector<double> y_pred = {0.8, 2.2, 2.7};
    
    auto gradient = Math::negativeGradient(y_true, y_pred);
    
    EXPECT_EQ(gradient.size(), y_true.size());
    EXPECT_NEAR(gradient[0], 0.2, 0.001);
    EXPECT_NEAR(gradient[1], -0.2, 0.001);
    EXPECT_NEAR(gradient[2], 0.3, 0.001);
}

// ====== Feature Tests ======
TEST(GetMaxFeatureTests, NormalValues) {
    Matrix values = {
        {3, -1, 2},
        {1, 5, 0},
        {4, 0, -3}
    };
    EXPECT_EQ(getMaxFeature(values, 0), 4);
    EXPECT_EQ(getMaxFeature(values, 1), 5);
    EXPECT_EQ(getMaxFeature(values, 2), 2);
}

TEST(GetMaxFeatureTests, EmptyMatrix) {
    Matrix values = {};
    EXPECT_THROW(getMaxFeature(values, 0), std::out_of_range);
}

TEST(GetMaxFeatureTests, FeatureIndexOutOfBounds) {
    Matrix values = {{1, 2, 3}, {4, 5, 6}};
    EXPECT_THROW(getMaxFeature(values, 3), std::out_of_range);
    EXPECT_THROW(getMaxFeature(values, -1), std::out_of_range);
}

TEST(GetMinFeatureTests, NormalValues) {
    Matrix values = {
        {3, -1, 2},
        {1, 5, 0},
        {4, 0, -3}
    };
    EXPECT_EQ(getMinFeature(values, 0), 1);
    EXPECT_EQ(getMinFeature(values, 1), -1);
    EXPECT_EQ(getMinFeature(values, 2), -3);
}

TEST(GetMinFeatureTests, EmptyMatrix) {
    Matrix values = {};
    EXPECT_THROW(getMinFeature(values, 0), std::out_of_range);
}

TEST(GetMeanFeatureTests, NormalValues) {
    Matrix values = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    EXPECT_FLOAT_EQ(getMeanFeature(values, 0), 4.0f);
    EXPECT_FLOAT_EQ(getMeanFeature(values, 1), 5.0f);
    EXPECT_FLOAT_EQ(getMeanFeature(values, 2), 6.0f);
}

// ====== Splitting Criteria Tests ======
TEST(SplittingCriteriaTest, MeanSquaredErrorTest) {
    MeanSquaredError mse;
    std::vector<std::vector<double>> data = {{1.0}, {2.0}, {3.0}};
    std::vector<double> labels = {2.0, 4.0, 6.0};
    double error = mse.calculate(data, labels, 0);
    EXPECT_GT(error, 0.0);
}

// ====== DrawUniqueNumbers Tests ======
TEST(DrawUniqueNumbersTests, HandlesMoreThanAvailableRows) {
    std::vector<int> result = drawUniqueNumbers(5, 3);
    EXPECT_EQ(result.size(), 3);
}

TEST(DrawUniqueNumbersTests, HandlesExactRows) {
    int n = 3;
    int rows = 3;
    std::vector<int> result = drawUniqueNumbers(n, rows);
    EXPECT_EQ(result.size(), n);
    for (int value : result) {
        EXPECT_GE(value, 0);
        EXPECT_LT(value, rows);
        EXPECT_EQ(std::count(result.begin(), result.end(), value), 1);
    }
}

// ====== Threshold Tests ======
TEST(CompareThresholdsTest, ReturnsMinVarianceThreshold) {
    std::vector<Threshold> thresholds = {
        Threshold(0, 10, 5.0),
        Threshold(1, 20, 2.5),
        Threshold(2, 15, 7.0),
        Threshold(0, 30, 4.0)
    };
    Threshold best_threshold = compareThresholds(thresholds);
    EXPECT_EQ(best_threshold.feature_index, 1);
    EXPECT_EQ(best_threshold.value, 20);
    EXPECT_FLOAT_EQ(best_threshold.weighted_variance, 2.5);
}

TEST(CompareThresholdsTest, HandlesSameVariancesThreshold) {
    std::vector<Threshold> thresholds = {
        Threshold(0, 10, 3.5),
        Threshold(1, 20, 3.5),
        Threshold(2, 15, 3.5),
    };
    Threshold best_threshold = compareThresholds(thresholds);
    EXPECT_EQ(best_threshold.feature_index, 0);
    EXPECT_EQ(best_threshold.value, 10);
    EXPECT_FLOAT_EQ(best_threshold.weighted_variance, 3.5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 