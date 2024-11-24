#include <gtest/gtest.h>
#include "boosting.h"
#include "loss_function.h"

// ====== Boosting Test Class ======
class BoostingTest : public ::testing::Test {
protected:
    MeanSquaredError mse;
    std::unique_ptr<LossFunction> loss_function;
    Boosting* boosting;

    void SetUp() override {
        loss_function = std::make_unique<LeastSquaresLoss>();
        boosting = new Boosting(5, 3, 0.1, &mse, std::move(loss_function));
    }

    void TearDown() override {
        delete boosting;
    }
};
// ====== Boosting Tests ======
TEST_F(BoostingTest, InitializePredictionTest) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}
    };
    std::vector<double> labels = {2.0, 4.0, 6.0};

    boosting->train(data, labels);
    auto predictions = boosting->predict(data);

    EXPECT_EQ(predictions.size(), data.size());
    for (const auto& pred : predictions) {
        EXPECT_GE(pred, 0.0);
    }
}
// ====== Train and Predict Tests ======
TEST_F(BoostingTest, TrainAndPredictTest) {
    std::vector<std::vector<double>> train_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}
    };
    std::vector<double> train_labels = {2.0, 4.0, 6.0, 8.0};

    boosting->train(train_data, train_labels);
    
    std::vector<std::vector<double>> test_data = {{2.5, 2.5}};
    auto predictions = boosting->predict(test_data);

    EXPECT_EQ(predictions.size(), 1);
    EXPECT_GE(predictions[0], 0.0);
}
// ====== Evaluate Tests ====== 
TEST_F(BoostingTest, EvaluateTest) {
    std::vector<std::vector<double>> train_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}
    };
    std::vector<double> train_labels = {2.0, 4.0, 6.0, 8.0};

    std::vector<std::vector<double>> test_data = {
        {1.5, 1.5}, {2.5, 2.5}, {3.5, 3.5}
    };
    std::vector<double> test_labels = {3.0, 5.0, 7.0};

    boosting->train(train_data, train_labels);
    double mse = boosting->evaluate(test_data, test_labels);

    EXPECT_GE(mse, 0.0);
} 