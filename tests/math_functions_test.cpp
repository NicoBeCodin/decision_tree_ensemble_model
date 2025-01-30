#include <gtest/gtest.h>
#include "../src/functions/tree/math_functions.h"

TEST(MathFunctionsTest, CalculateMeanTest) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(Math::calculateMean(values), 3.0);
}

TEST(MathFunctionsTest, CalculateMSETest) {
    std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 6.0};
    EXPECT_DOUBLE_EQ(Math::calculateMSE(values), 1.6);
}

TEST(MathFunctionsTest, EmptyVectorTest) {
    std::vector<double> empty_values;
    EXPECT_DOUBLE_EQ(Math::calculateMean(empty_values), 0.0);
    EXPECT_DOUBLE_EQ(Math::calculateMSE(empty_values), 0.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 