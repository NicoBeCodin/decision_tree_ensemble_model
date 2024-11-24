#include <gtest/gtest.h>
#include "data_clean.h"

class DataCleanTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initial setup if needed
    }
};

// Add your tests here
TEST_F(DataCleanTest, BasicTest) {
    // Your first test
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
