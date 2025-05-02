#pragma once

#include <vector>
#include <string>
#include <LightGBM/c_api.h>

/**
 * A simple wrapper for LightGBM regression using C API.
 */
class MyLightGBM {
public:
    MyLightGBM();
    ~MyLightGBM();

    // Train the model
    void train(const std::vector<std::vector<float>>& X,
        const std::vector<float>& y,
        const std::string& params,
        int n_iters,
        int num_threads = 1);

    // Predict using flattened input (row-major order)
    std::vector<double> predict(const std::vector<float>& data,
                                int n_samples,
                                int n_features);

    // Get feature importance scores
    std::vector<double> featureImportance(int importance_type = C_API_FEATURE_IMPORTANCE_GAIN);

    // Save the model to a file
    void saveModel(const std::string& filename);

    // Getters
    int numFeatures() const { return n_features_; }
    int numSamples() const { return n_samples_; }

private:
    BoosterHandle booster_;
    DatasetHandle train_data_;
    int n_features_;
    int n_samples_;
    bool trained_;
};
