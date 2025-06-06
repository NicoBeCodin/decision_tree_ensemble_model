#ifndef BOOSTING_IMPROVED_H
#define BOOSTING_IMPROVED_H

#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <memory>  
#include <unordered_map>
#include <omp.h>
#include <string>
#include <fstream>
#include <map>
#include <iostream>
#include <functional>
#include "binning_methods.h"

#define MAX_BINS 2048          
#define MIN_SAMPLES_FOR_PARALLEL 1000

class MemoryPool {
    public:
        MemoryPool() = default;
        ~MemoryPool() = default;
    
        void init(std::size_t max_vectors, std::size_t /*max_samples*/) {
            next_ = 0;
            pool_.clear();
            pool_.reserve(max_vectors);
            for (std::size_t i = 0; i < max_vectors; ++i)
                pool_.emplace_back(std::make_unique<std::vector<int>>());
        }
    
        std::vector<int>& get_vector() {
            if (next_ >= pool_.size()) {
                static std::vector<int> dummy;
                dummy.clear();
                return dummy; // fallback
            }
            auto* v = pool_[next_++].get();
            v->clear();
            return *v;
        }
    
        void reset() {
            next_ = 0;
        }
    
    private:
        std::vector<std::unique_ptr<std::vector<int>>> pool_;
        std::size_t next_ = 0;
    };

class ImprovedGBDT {
public:
    // Binning method enum
    enum BinningMethod {
        NONE,
        QUANTILE,
        FREQUENCY
    };

    // Node structure with cache line alignment
    struct alignas(64) Node {
        bool is_leaf;
        int split_feature;
        double split_value;
        double leaf_value;
        Node* left;
        Node* right;
        double sum_grad;
        double sum_hess;
        int sample_count;
        
        Node(): is_leaf(false), split_feature(-1), split_value(0.0), leaf_value(0.0), 
                left(nullptr), right(nullptr), sum_grad(0.0), sum_hess(0.0), sample_count(0) {}
    };

    struct Tree {
        Node* root;
        Tree(): root(nullptr) {}
    };

    // Constructor with improved parameter organization
    ImprovedGBDT(int n_estimators, 
                 int max_depth, 
                 double learning_rate,
                 bool useDart, 
                 double drop_rate, 
                 double skip_rate,
                 BinningMethod binning_method, 
                 int num_bins,
                 int min_samples_leaf,
                 double l2_reg,
                 double feature_sample_ratio,
                 int early_stopping_rounds,
                 int num_threads)
        : n_estimators(n_estimators), 
          max_depth(max_depth), 
          learning_rate(learning_rate),
          useDart(useDart), 
          drop_rate(drop_rate), 
          skip_rate(skip_rate),
          binning_method(binning_method), 
          num_bins(num_bins),
          min_samples_leaf(min_samples_leaf),
          l2_reg(l2_reg),
          feature_sample_ratio(feature_sample_ratio),
          early_stopping_rounds(early_stopping_rounds),
          num_threads(num_threads) {
        rng.seed(123);  // Fixed seed for reproducibility
        initial_prediction = 0.0;
    }

    // Destructor to free memory
    ~ImprovedGBDT() {
        for (auto &tree : trees) {
            freeTree(tree.root);
        }
    }
    
    // Histogram entry with cache line alignment
    struct alignas(64) HistogramEntry {
        double grad_sum;
        double hess_sum;
        int count;
        HistogramEntry() : grad_sum(0.0), hess_sum(0.0), count(0) {}
    };


    // Core training method
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<double>& y,
             const std::vector<std::vector<double>>* X_val = nullptr,
             const std::vector<double>* y_val = nullptr);

    // Single sample prediction
    double predict(const std::vector<double>& x) const;

    // Batch prediction
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    
    // Calculate feature importance
    std::vector<double> featureImportance() const;
    
    // Model serialization
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);
    
    // Get training parameters as a map
    std::map<std::string, std::string> getTrainingParameters() const;
    
    // Get a readable string of training parameters
    std::string getTrainingParametersString() const;

    // Get initial prediction
    double getInitialPrediction() const { return initial_prediction; }

private:
    // Model parameters
    int n_estimators;
    int max_depth;
    double learning_rate;
    bool useDart;
    double drop_rate;
    double skip_rate;
    BinningMethod binning_method;
    int num_bins;
    int min_samples_leaf;  // New: minimum samples in leaf
    double l2_reg;         // Configurable L2 regularization
    double feature_sample_ratio; // Feature sampling ratio
    int early_stopping_rounds;  // Early stopping rounds
    MemoryPool memory_pool;
    
    // Model state
    double initial_prediction;
    std::vector<Tree> trees;
    std::vector<double> tree_weights;
    std::vector<double> y_pred_train;
    std::default_random_engine rng;
    int num_threads;
    int n_features;  // Store feature count
    
    // Validation tracking
    double best_val_score;
    int best_iteration;
    int current_round;

    // Binning objects
    std::unique_ptr<Binning::QuantileSketch> quantile_binner;
    std::unique_ptr<Binning::FrequencyBinning> frequency_binner;
    
    // Feature binning cache
    struct FeatureBinCache {
        std::vector<std::vector<int>> feature_bins;
    };
    FeatureBinCache bin_cache;
    
    // Recursive tree building methods
    Node* buildTreeRecursive(
        const std::vector<std::vector<double>>& X, 
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices, 
        const std::vector<int>& feature_indices,
        int depth,
        double sum_gradients,
        double sum_hessians);
    
    Node* buildTreeRecursiveBinned(
        const std::vector<std::vector<double>>& X, 
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices, 
        const std::vector<int>& feature_indices,
        int depth,
        double sum_gradients,
        double sum_hessians,
        std::shared_ptr<const std::vector<std::vector<HistogramEntry>>> parent_hist = nullptr);
    
    // Get bin index, optimized for inlining
    inline int getBinIndex(double value, int feature_idx, int sample_idx = -1) const {
        // Use cache if valid
        if (sample_idx >= 0 && feature_idx < (int)bin_cache.feature_bins.size() && 
            sample_idx < (int)bin_cache.feature_bins[feature_idx].size()) {
            return bin_cache.feature_bins[feature_idx][sample_idx];
        }
        
        // Handle NaN values
        if (std::isnan(value)) return 0;
        
        // Compute bin index
        if (binning_method == QUANTILE && quantile_binner) {
            return quantile_binner->getBin(value, feature_idx);
        } else if (binning_method == FREQUENCY && frequency_binner) {
            return frequency_binner->getBin(value, feature_idx);
        }
        return 0;
    }
    
    // Get split value from bin index
    inline double getSplitValueFromBin(int feature_idx, int bin_idx) const {
        if (binning_method == QUANTILE && quantile_binner) {
            return quantile_binner->getSplitValue(feature_idx, bin_idx);
        } else if (binning_method == FREQUENCY && frequency_binner) {
            return frequency_binner->getSplitValue(feature_idx, bin_idx);
        }
        return 0.0;
    }
    
    // Calculate optimal leaf value
    inline double calculateLeafValue(double sum_gradients, double sum_hessians) const {
        // More robust handling of small hessian values
        if (std::abs(sum_hessians) < 1e-10) {
            if (std::abs(sum_gradients) < 1e-10) {
                return 0.0;
            }
            // Limit to a reasonable value when hessian is near zero
            return -std::copysign(std::min(std::abs(sum_gradients), 100.0), sum_gradients);
        }
        
        // Apply regularization and limit extreme values
        double raw_value = -sum_gradients / (sum_hessians + l2_reg);
        
        // Limit leaf values to prevent extreme predictions
        const double max_leaf_value = 100.0;
        return std::max(-max_leaf_value, std::min(max_leaf_value, raw_value));
    }
    
    inline double calculateSplitGain(double left_grad, double left_hess,
                               double right_grad, double right_hess,
                               double parent_grad, double parent_hess) const {
    // Check for minimum hessian values to ensure numerical stability
    if (left_hess < 1e-10 || right_hess < 1e-10) {
        return 0.0;
    }
    
    // Calculate scores with regularization
    double left_score = left_grad * left_grad / (left_hess + l2_reg);
    double right_score = right_grad * right_grad / (right_hess + l2_reg);
    double parent_score = parent_grad * parent_grad / (parent_hess + l2_reg);
    
    // Calculate gain
    double gain = left_score + right_score - parent_score;
    
    // Add a complexity penalty based on number of samples
    // This slightly penalizes splits with very unbalanced distributions
    double left_weight = left_hess / (left_hess + right_hess);
    double right_weight = right_hess / (left_hess + right_hess);
    double balance_factor = 4.0 * left_weight * right_weight; // Peaks at 1.0 when balanced
    
    // Apply minimal regularization to prefer balanced splits slightly
    gain *= (0.9 + 0.1 * balance_factor);
    
    // Prevent negative gains (shouldn't happen with proper calculation)
    return std::max(0.0, gain);
    }
    
    // Methods for building and optimizing histograms
    void precomputeFeatureBins(const std::vector<std::vector<double>>& X);
    
    void buildHistogram(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& indices,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        std::vector<std::vector<HistogramEntry>>& hist,
        const std::vector<int>& feature_indices);
        
    bool findBestSplit(
        const std::vector<std::vector<HistogramEntry>>& hist,
        double sum_gradients,
        double sum_hessians,
        int& best_feature,
        int& best_bin,
        double& best_gain,
        const std::vector<int>& feature_indices);
        
    void splitNodeHistogram(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& indices,
        int best_feature,
        int best_bin,
        double best_split_value,
        std::vector<int>& left_indices,
        std::vector<int>& right_indices,
        double& left_grad_sum,
        double& left_hess_sum,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians);

    // Helper method to select trees for dropout in DART
    std::vector<int> selectDropoutTrees();
    
    // Free tree memory
    void freeTree(Node* node) {
        if (!node) return;
        freeTree(node->left);
        freeTree(node->right);
        delete node;
    }
    
    // Serialize a node for model saving
    void serializeNode(const Node* node, std::ostream& out) const;
    
    // Deserialize a node for model loading
    Node* deserializeNode(std::istream& in);
};

#endif // BOOSTING_IMPROVED_H