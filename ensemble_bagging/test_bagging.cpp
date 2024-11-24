#include <gtest/gtest.h>
#include "bagging.h"
#include "../functions_tree/splitting_criteria.h"
// ====== Bagging Test Class ======
class BaggingTest : public ::testing::Test {
protected:
    MeanSquaredError mse;
    Bagging* bagging;

    void SetUp() override {
        bagging = new Bagging(5, 3, &mse);
    }

    void TearDown() override {
        delete bagging;
    }
};
// ====== Bagging Tests ======
TEST_F(BaggingTest, BootstrapSampleTest) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}
    };
    std::vector<double> labels = {10.0, 20.0, 30.0, 40.0};

    auto [sampled_data, sampled_labels] = bagging->bootstrapSample(data, labels);

    EXPECT_EQ(sampled_data.size(), data.size());
    EXPECT_EQ(sampled_labels.size(), labels.size());
}
// ====== Prediction Tests ======
TEST_F(BaggingTest, PredictionTest) {
    std::vector<std::vector<double>> train_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}
    };
    std::vector<double> train_labels = {2.0, 4.0, 6.0, 8.0};

    bagging->train(train_data, train_labels);

    std::vector<double> test_sample = {2.5, 2.5};
    double prediction = bagging->predict(test_sample);

    EXPECT_GE(prediction, 0.0);
    EXPECT_LE(prediction, 10.0);
}
// ====== Evaluate Tests ====== 
TEST_F(BaggingTest, EvaluateTest) {
    std::vector<std::vector<double>> train_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}
    };
    std::vector<double> train_labels = {2.0, 4.0, 6.0, 8.0};

    std::vector<std::vector<double>> test_data = {
        {1.5, 1.5}, {2.5, 2.5}, {3.5, 3.5}
    };
    std::vector<double> test_labels = {3.0, 5.0, 7.0};

    bagging->train(train_data, train_labels);
    double mse = bagging->evaluate(test_data, test_labels);

    EXPECT_GE(mse, 0.0);
} 