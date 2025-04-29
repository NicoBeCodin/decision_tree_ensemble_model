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
#include "binning_methods.h"


#define MAX_BINS 2048          
#define MIN_SAMPLES_FOR_PARALLEL 1000  
#define CACHE_LINE_SIZE 64       

class ImprovedGBDT {
public:
    
    enum BinningMethod {
        NONE,           
        QUANTILE,       // Quantile method (XGBoost style)
        FREQUENCY       // Frequency binning (LightGBM style)
    };

    // Cache line aligned node structure, avoids false sharing
    struct alignas(CACHE_LINE_SIZE) Node {
        bool is_leaf;
        int split_feature;
        double split_value;
        double leaf_value;
        Node* left;
        Node* right;
        // Stats cache to avoid recalculation
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

    // Constructor with unchanged parameters
    ImprovedGBDT(int n_estimators, int max_depth, double learning_rate = 0.1,
                 bool useDart = false, double drop_rate = 0.1, double skip_rate = 0.0,
                 BinningMethod binning_method = NONE, int num_bins = 256)
        : n_estimators(n_estimators), max_depth(max_depth), learning_rate(learning_rate),
          useDart(useDart), drop_rate(drop_rate), skip_rate(skip_rate),
          binning_method(binning_method), num_bins(num_bins) {
        rng.seed(123);  // Fixed seed for reproducible results
        initial_prediction = 0.0;
    }

    // Destructor to free memory
    ~ImprovedGBDT() {
        for (auto &tree : trees) {
            freeTree(tree.root);
        }
    }

    // Train method interface remains unchanged
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    // Single sample prediction interface remains unchanged
    double predict(const std::vector<double>& x) const;

    // Batch prediction interface remains unchanged
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    int n_estimators;
    int max_depth;
    double learning_rate;
    bool useDart;
    double drop_rate;
    double skip_rate;
    BinningMethod binning_method;
    int num_bins;
    double initial_prediction;
    std::vector<Tree> trees;
    std::vector<double> tree_weights;
    std::vector<double> y_pred_train;
    std::default_random_engine rng;
    int num_threads;  // Cache available threads

    // Binning objects
    std::unique_ptr<Binning::QuantileSketch> quantile_binner;
    std::unique_ptr<Binning::FrequencyBinning> frequency_binner;
    
    // Optimized cache structure
    struct alignas(CACHE_LINE_SIZE) BinCacheEntry {
        int bin_idx;
        bool valid;
        BinCacheEntry() : bin_idx(0), valid(false) {}
    };
    
    struct FeatureBinCache {
        std::vector<BinCacheEntry> entries;
        std::vector<std::vector<int>> feature_bins;
    };
    
    // Feature binning cache
    FeatureBinCache bin_cache;
    
    // Histogram struct, aligned to cache line
    struct alignas(CACHE_LINE_SIZE) HistogramEntry {
        double grad_sum;
        double hess_sum;
        int count;
        HistogramEntry() : grad_sum(0.0), hess_sum(0.0), count(0) {}
    };
    
    // Tree building memory pool
    struct MemoryPool {
        std::vector<std::vector<int>> index_vectors;
        size_t current_index = 0;
        
        void init(int n_vectors, int capacity) {
            index_vectors.resize(n_vectors);
            for (auto& vec : index_vectors) {
                vec.reserve(capacity);
            }
            current_index = 0;
        }
        
        std::vector<int>& get_vector() {
            if (current_index >= index_vectors.size()) {
                // Add new vector if exhausted
                index_vectors.emplace_back();
                index_vectors.back().reserve(index_vectors[0].capacity());
            }
            index_vectors[current_index].clear();
            return index_vectors[current_index++];
        }
        
        void reset() {
            current_index = 0;
        }
    };
    
    MemoryPool memory_pool;

    // Highly optimized recursive tree building
    Node* buildTreeRecursive(
        const std::vector<std::vector<double>>& X, 
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices, 
        int depth,
        double sum_gradients,
        double sum_hessians);
    
    // Optimized tree building with binning
    Node* buildTreeRecursiveBinned(
        const std::vector<std::vector<double>>& X, 
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices, 
        int depth,
        double sum_gradients,
        double sum_hessians,
        const std::vector<std::vector<HistogramEntry>>* parent_hist = nullptr);
    
    // Optimized bin index lookup, inlined to reduce function call overhead
    inline int getBinIndex(double value, int feature_idx, int sample_idx = -1) const {
        // Return cache if valid
        if (sample_idx >= 0 && feature_idx < (int)bin_cache.feature_bins.size() && 
            sample_idx < (int)bin_cache.feature_bins[feature_idx].size()) {
            return bin_cache.feature_bins[feature_idx][sample_idx];
        }
        
        // Otherwise compute bin index
        if (binning_method == QUANTILE && quantile_binner) {
            return quantile_binner->getBin(value, feature_idx);
        } else if (binning_method == FREQUENCY && frequency_binner) {
            return frequency_binner->getBin(value, feature_idx);
        }
        return 0; // Default
    }
    
    // Get split value from bin index, inlined
    inline double getSplitValueFromBin(int feature_idx, int bin_idx) const {
        if (binning_method == QUANTILE && quantile_binner) {
            return quantile_binner->getSplitValue(feature_idx, bin_idx);
        } else if (binning_method == FREQUENCY && frequency_binner) {
            return frequency_binner->getSplitValue(feature_idx, bin_idx);
        }
        return 0.0; // Default
    }
    
    // Calculate optimal leaf value, inlined
    inline double calculateLeafValue(double sum_gradients, double sum_hessians, double l2_reg = 1.0) const {
        // Avoid division by near-zero
        if (std::abs(sum_hessians) < 1e-10) {
            return 0.0;
        }
        return -sum_gradients / (sum_hessians + l2_reg);
    }
    
    // Calculate split gain, inlined
    inline double calculateSplitGain(double left_grad, double left_hess,
                                   double right_grad, double right_hess,
                                   double parent_grad, double parent_hess,
                                   double l2_reg = 1.0) const {
        // Safety check
        if (left_hess < 1e-10 || right_hess < 1e-10) {
            return 0.0;
        }
        
        // Efficient gain calculation
        double left_score = left_grad * left_grad / (left_hess + l2_reg);
        double right_score = right_grad * right_grad / (right_hess + l2_reg);
        double parent_score = parent_grad * parent_grad / (parent_hess + l2_reg);
        
        return left_score + right_score - parent_score;
    }
    
    // Precompute bins for all samples and features
    void precomputeFeatureBins(const std::vector<std::vector<double>>& X);
    
    // Build histogram in parallel
    void buildHistogram(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& indices,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        std::vector<std::vector<HistogramEntry>>& hist);
        
    // Find best split
    bool findBestSplit(
        const std::vector<std::vector<HistogramEntry>>& hist,
        double sum_gradients,
        double sum_hessians,
        int& best_feature,
        int& best_bin,
        double& best_gain);
        
    // Split node using histogram
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

    // Free tree memory
    void freeTree(Node* node) {
        if (!node) return;
        freeTree(node->left);
        freeTree(node->right);
        delete node;
    }
};

#endif // BOOSTING_IMPROVED_H