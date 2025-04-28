#include "my_lightgbm.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

extern "C" const char* LGBM_GetLastError();

MyLightGBM::MyLightGBM()
    : booster_(nullptr), n_features_(0), n_samples_(0),
      train_data_(nullptr), trained_(false) {}

MyLightGBM::~MyLightGBM() {
    if (booster_) {
        LGBM_BoosterFree(booster_);
        booster_ = nullptr;
    }
    if (train_data_) {
        LGBM_DatasetFree(train_data_);
        train_data_ = nullptr;
    }
}

void MyLightGBM::train(const std::vector<std::vector<float>>& X,
                       const std::vector<float>& y,
                       const std::string& params,
                       int n_iters) {
    n_samples_ = static_cast<int>(X.size());
    if (n_samples_ == 0) throw std::runtime_error("No training data.");
    n_features_ = static_cast<int>(X[0].size());
    for (int i = 1; i < n_samples_; ++i) {
        if (static_cast<int>(X[i].size()) != n_features_)
            throw std::runtime_error("Inconsistent feature dimensions.");
    }
    if (static_cast<int>(y.size()) != n_samples_)
        throw std::runtime_error("Mismatched labels size.");

    std::vector<float> data_flat(n_samples_ * n_features_);
    for (int i = 0; i < n_samples_; ++i)
        for (int j = 0; j < n_features_; ++j)
            data_flat[i * n_features_ + j] = X[i][j];

    int ret = LGBM_DatasetCreateFromMat(
        (void*)data_flat.data(),
        C_API_DTYPE_FLOAT32,
        n_samples_,
        n_features_,
        1,
        params.c_str(),
        nullptr,
        &train_data_
    );
    if (ret != 0) throw std::runtime_error(LGBM_GetLastError());

    ret = LGBM_DatasetSetField(
        train_data_, "label",
        (void*)y.data(),
        n_samples_,
        C_API_DTYPE_FLOAT32
    );
    if (ret != 0) throw std::runtime_error(LGBM_GetLastError());

    ret = LGBM_BoosterCreate(
        train_data_,
        params.c_str(),
        &booster_
    );
    if (ret != 0) throw std::runtime_error(LGBM_GetLastError());

    for (int iter = 0; iter < n_iters; ++iter) {
        int is_finished = 0;
        ret = LGBM_BoosterUpdateOneIter(booster_, &is_finished);
        if (ret != 0) throw std::runtime_error(LGBM_GetLastError());
    }

    trained_ = true;
}

std::vector<double> MyLightGBM::predict(const std::vector<float>& data, int n_samples, int n_features) {
    if (data.empty() || n_samples <= 0 || n_features <= 0 || booster_ == nullptr)
        return {};

    if ((size_t)n_samples * n_features != data.size()) {
        std::cerr << "Error: Dimension mismatch in input data." << std::endl;
        return {};
    }

    int num_classes = 1;
    if (LGBM_BoosterGetNumClasses(booster_, &num_classes) != 0) {
        std::cerr << "Error: " << LGBM_GetLastError() << std::endl;
        return {};
    }

    std::vector<double> preds((size_t)n_samples * num_classes);
    int64_t out_len = 0;
    int ret = LGBM_BoosterPredictForMat(
        booster_,
        data.data(),
        C_API_DTYPE_FLOAT32,
        n_samples,
        n_features,
        1,
        C_API_PREDICT_NORMAL,
        0,
        -1,
        "",
        &out_len,
        preds.data()
    );
    if (ret != 0) {
        std::cerr << "Error: " << LGBM_GetLastError() << std::endl;
        return {};
    }

    return preds;
}

std::vector<double> MyLightGBM::featureImportance(int importance_type) {
    std::vector<double> importances;
    if (booster_ == nullptr) {
        std::cerr << "Error: Booster not initialized." << std::endl;
        return importances;
    }

    int num_feature = 0;
    if (LGBM_BoosterGetNumFeature(booster_, &num_feature) != 0) {
        std::cerr << "Error: " << LGBM_GetLastError() << std::endl;
        return importances;
    }

    importances.resize(num_feature);
    if (LGBM_BoosterFeatureImportance(booster_, 0, importance_type, importances.data()) != 0) {
        std::cerr << "Error: " << LGBM_GetLastError() << std::endl;
        importances.clear();
    }
    return importances;
}

void MyLightGBM::saveModel(const std::string& filename) {
    if (!booster_) throw std::runtime_error("Model not trained.");
    int ret = LGBM_BoosterSaveModel(
        booster_,
        0,
        -1,
        0,
        filename.c_str()
    );
    if (ret != 0)
        throw std::runtime_error(std::string("LGBM_BoosterSaveModel failed: ") + (LGBM_GetLastError() ? LGBM_GetLastError() : "Unknown error"));
}
