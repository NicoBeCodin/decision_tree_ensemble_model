#include "boosting_improved.h"
#include <omp.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <ctime>
#include <map>
#include <iomanip>

// Method to select trees for dropout in DART with proper sampling
std::vector<int> ImprovedGBDT::selectDropoutTrees() {
    if (trees.empty() || drop_rate <= 0.0) return {};
    
    const int total_trees = trees.size();
    
    // Determine how many trees to drop (ensure it's at least 1 but not all)
    int drop_count = std::max(1, std::min(total_trees - 1, 
                            static_cast<int>(std::round(drop_rate * total_trees))));
    
    // Create indices of all trees
    std::vector<int> all_indices(total_trees);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    // Randomly shuffle
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    
    // Take first drop_count elements
    return std::vector<int>(all_indices.begin(), all_indices.begin() + drop_count);
}

void ImprovedGBDT::precomputeFeatureBins(const std::vector<std::vector<double>>& X) {
    if (binning_method == NONE || X.empty()) return;
    
    int n_samples = X.size();
    bin_cache.feature_bins.resize(n_features);
    
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < n_features; ++f) {
        bin_cache.feature_bins[f].resize(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            // Handle NaN values
            if (std::isnan(X[i][f])) {
                bin_cache.feature_bins[f][i] = 0;
                continue;
            }
            
            if (binning_method == QUANTILE && quantile_binner) {
                bin_cache.feature_bins[f][i] = quantile_binner->getBin(X[i][f], f);
            } else if (binning_method == FREQUENCY && frequency_binner) {
                bin_cache.feature_bins[f][i] = frequency_binner->getBin(X[i][f], f);
            }
        }
    }
}

void ImprovedGBDT::buildHistogram(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& indices,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    std::vector<std::vector<HistogramEntry>>& hist,
    const std::vector<int>& feature_indices) {

    const int max_threads = omp_get_max_threads();
    const int n_feats     = n_features;
    const int n_bins      = num_bins;

    // 1. 函数级 static，所有线程共享这块缓冲区
    std::vector<std::vector<HistogramEntry>> thread_hist(
        max_threads,
        std::vector<HistogramEntry>(n_feats * (n_bins + 1))
    );

    // 2. 清空主直方图
    for (int f : feature_indices) {
        std::fill(
            hist[f].begin(),
            hist[f].end(),
            HistogramEntry()
        );
    }

   
    const size_t N = indices.size();
    const size_t L = feature_indices.size();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &local = thread_hist[tid];

        
        for (size_t i_f = 0; i_f < L; ++i_f) {
            int f = feature_indices[i_f];
            HistogramEntry *ptr = &local[f * (n_bins + 1)];
            for (int b = 0; b <= n_bins; ++b) {
                ptr[b] = HistogramEntry();
            }
        }

     
        #pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            int sample = indices[i];
            for (size_t i_f = 0; i_f < L; ++i_f) {
                int f = feature_indices[i_f];
                int bin = getBinIndex(X[sample][f], f, sample);
                auto &cell = local[f * (n_bins + 1) + bin];
                cell.grad_sum += gradients[sample];
                cell.hess_sum += hessians[sample];
                cell.count    += 1;
            }
        }

        
        #pragma omp barrier

        
        #pragma omp for schedule(dynamic)
        for (size_t i_f = 0; i_f < L; ++i_f) {
            int f = feature_indices[i_f];
            HistogramEntry *main_ptr = hist[f].data();
            for (int b = 0; b <= n_bins; ++b) {
                double g = 0.0, h = 0.0;
                int    c = 0;
                for (int t = 0; t < max_threads; ++t) {
                    auto &e = thread_hist[t][f * (n_bins + 1) + b];
                    g += e.grad_sum; h += e.hess_sum; c += e.count;
                }
                main_ptr[b].grad_sum = g;
                main_ptr[b].hess_sum = h;
                main_ptr[b].count    = c;
            }
        }
    } 

    
    double sum_g = 0.0, sum_h = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum_g += gradients[indices[i]];
        sum_h += hessians[indices[i]];
    }
    for (int f : feature_indices) {
        hist[f][n_bins].grad_sum = sum_g;
        hist[f][n_bins].hess_sum = sum_h;
        hist[f][n_bins].count    = (int)N;
    }
}


bool ImprovedGBDT::findBestSplit(
    const std::vector<std::vector<HistogramEntry>>& hist,
    double sum_gradients,
    double sum_hessians,
    int& best_feature,
    int& best_bin,
    double& best_gain,
    const std::vector<int>& feature_indices) {
    
    best_feature = -1;
    best_bin = 0;
    best_gain = 0.0;
    
    #pragma omp parallel
    {
        double local_best_gain = 0.0;
        int local_best_feature = -1;
        int local_best_bin = 0;
        
        #pragma omp for schedule(dynamic)
        for (size_t fi = 0; fi < feature_indices.size(); ++fi) {
            int f = feature_indices[fi];
            double left_grad = 0.0;
            double left_hess = 0.0;
            int left_count = 0;
            
            for (int bin = 0; bin < num_bins; ++bin) {
                if (hist[f][bin].count == 0) continue;
                
                left_grad += hist[f][bin].grad_sum;
                left_hess += hist[f][bin].hess_sum;
                left_count += hist[f][bin].count;
                
                double right_grad = sum_gradients - left_grad;
                double right_hess = sum_hessians - left_hess;
                int right_count = hist[f][num_bins].count - left_count;
                
                // Check min_samples_leaf constraint
                if (left_count < min_samples_leaf || right_count < min_samples_leaf) continue;
                
                double gain = calculateSplitGain(
                    left_grad, left_hess, 
                    right_grad, right_hess, 
                    sum_gradients, sum_hessians
                );
                
                if (gain > local_best_gain) {
                    local_best_gain = gain;
                    local_best_feature = f;
                    local_best_bin = bin;
                }
            }
        }
        
        #pragma omp critical
        {
            if (local_best_gain > best_gain) {
                best_gain = local_best_gain;
                best_feature = local_best_feature;
                best_bin = local_best_bin;
            }
        }
    }
    
    return best_feature != -1;
}

void ImprovedGBDT::splitNodeHistogram(
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
    const std::vector<double>& hessians) {
    
    left_indices.clear();
    right_indices.clear();
    left_grad_sum = 0.0;
    left_hess_sum = 0.0;
    
    const int n_samples = indices.size();
    left_indices.reserve(n_samples / 2);
    right_indices.reserve(n_samples / 2);
    
    if (n_samples > MIN_SAMPLES_FOR_PARALLEL) {
        // Parallel split for larger node samples
        std::vector<int> left_buffer(n_samples);
        std::vector<int> right_buffer(n_samples);
        std::vector<int> left_counts(omp_get_max_threads() + 1, 0);
        std::vector<int> right_counts(omp_get_max_threads() + 1, 0);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int left_count = 0;
            int right_count = 0;
            
            #pragma omp for schedule(static)
            for (int i = 0; i < n_samples; ++i) {
                int idx = indices[i];
                if (std::isnan(X[idx][best_feature])) {
                    // Send NaN values to left child by default
                    left_count++;
                } else if (X[idx][best_feature] <= best_split_value) {
                    left_count++;
                } else {
                    right_count++;
                }
            }
            
            left_counts[thread_id + 1] = left_count;
            right_counts[thread_id + 1] = right_count;
        }
        
        // Compute prefix sums
        for (int i = 1; i <= omp_get_max_threads(); ++i) {
            left_counts[i] += left_counts[i - 1];
            right_counts[i] += right_counts[i - 1];
        }
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int left_offset = left_counts[thread_id];
            int right_offset = right_counts[thread_id];
            
            #pragma omp for schedule(static) reduction(+:left_grad_sum,left_hess_sum)
            for (int i = 0; i < n_samples; ++i) {
                int idx = indices[i];
                if (std::isnan(X[idx][best_feature]) || X[idx][best_feature] <= best_split_value) {
                    left_buffer[left_offset++] = idx;
                    left_grad_sum += gradients[idx];
                    left_hess_sum += hessians[idx];
                } else {
                    right_buffer[right_offset++] = idx;
                }
            }
        }
        
        // Copy results to output vectors
        left_indices.assign(left_buffer.begin(), left_buffer.begin() + left_counts[omp_get_max_threads()]);
        right_indices.assign(right_buffer.begin(), right_buffer.begin() + right_counts[omp_get_max_threads()]);
    } else {
        // Serial split for smaller datasets
        for (int idx : indices) {
            if (std::isnan(X[idx][best_feature]) || X[idx][best_feature] <= best_split_value) {
                left_indices.push_back(idx);
                left_grad_sum += gradients[idx];
                left_hess_sum += hessians[idx];
            } else {
                right_indices.push_back(idx);
            }
        }
    }
}

void ImprovedGBDT::fit(const std::vector<std::vector<double>>& X, 
                       const std::vector<double>& y,
                       const std::vector<std::vector<double>>* X_val,
                       const std::vector<double>* y_val) {
    int n_samples = X.size();
    if (n_samples == 0) return;
    
    n_features = X[0].size();
    
    // Initialize validation tracking
    best_val_score = std::numeric_limits<double>::max();
    best_iteration = 0;
    current_round = 0;
    
    // Setup early stopping
    bool use_early_stopping = (early_stopping_rounds > 0 && X_val != nullptr && y_val != nullptr);
    
    // Initialize memory pool based on tree depth and sample count
    std::size_t max_vecs =
    static_cast<std::size_t>(num_threads) *
    static_cast<std::size_t>((1ULL << (max_depth + 1)) - 1);  // ≃ nœuds max
    memory_pool.init(max_vecs, n_samples);

    // Initialize binning methods
    if (binning_method == QUANTILE) {
        quantile_binner = std::make_unique<Binning::QuantileSketch>(num_bins);
        quantile_binner->build(X);
    } else if (binning_method == FREQUENCY) {
        frequency_binner = std::make_unique<Binning::FrequencyBinning>(num_bins);
        frequency_binner->build(X);
    }
    
    // Precompute feature bins if using binning
    if (binning_method != NONE) {
        precomputeFeatureBins(X);
    }

    // Calculate initial prediction (average of target values)
    double sum_y = 0.0;
    #pragma omp parallel for reduction(+:sum_y) schedule(static)
    for (int i = 0; i < n_samples; ++i) {
        sum_y += y[i];
    }
    initial_prediction = sum_y / n_samples;
    
    // Initialize predictions
    y_pred_train.assign(n_samples, initial_prediction);
    
    // Reserve space for trees and weights
    trees.reserve(n_estimators);
    tree_weights.reserve(n_estimators);
    
    // Create RNG for randomization
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Setup validation predictions if doing early stopping
    std::vector<double> y_pred_val;
    if (use_early_stopping) {
        y_pred_val.assign(X_val->size(), initial_prediction);
    }

    // Prepare gradients and hessians
    std::vector<double> gradients(n_samples);
    std::vector<double> hessians(n_samples, 1.0);

    // Main training loop
    std::vector<int> no_improvement_rounds;
    if (use_early_stopping) {
        no_improvement_rounds.resize(early_stopping_rounds, 0);
    }
    
    for (int iter = 0; iter < n_estimators; ++iter) {
        current_round = iter;
        
        // Skip logic for DART
        if (useDart && skip_rate > 0.0) {
            double skip_sample = dist(rng);
            if (skip_sample < skip_rate) {
                continue;
            }
        }

        // Compute predictions for gradient calculation
        std::vector<int> drop_indices;
        std::vector<double> pred_for_grad = y_pred_train;
        
        // DART: randomly drop trees when computing gradients
        if (useDart && !trees.empty()) {
            drop_indices = selectDropoutTrees();
            
            if (!drop_indices.empty()) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n_samples; ++i) {
                    for (int drop_idx : drop_indices) {
                        Node* node = trees[drop_idx].root;
                        while (node && !node->is_leaf) {
                            if (std::isnan(X[i][node->split_feature])) {
                                // Default direction for missing values
                                node = node->left;
                            } else {
                                node = (X[i][node->split_feature] <= node->split_value) ? 
                                        node->left : node->right;
                            }
                        }
                        if (node) {
                            pred_for_grad[i] -= tree_weights[drop_idx] * node->leaf_value;
                        }
                    }
                }
            }
        }

        // Compute gradients (MSE loss)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; ++i) {
            gradients[i] = pred_for_grad[i] - y[i];
            hessians[i] = 1.0;
        }

        // Prepare sample indices
        std::vector<int> all_indices(n_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        // Compute sum gradients and hessians
        double sum_gradients = 0.0;
        double sum_hessians = static_cast<double>(n_samples);
        
        #pragma omp parallel for reduction(+:sum_gradients) schedule(static)
        for (int i = 0; i < n_samples; ++i) {
            sum_gradients += gradients[i];
        }
        
        // Feature sampling
        std::vector<int> feature_indices(n_features);
        std::iota(feature_indices.begin(), feature_indices.end(), 0);
        
        if (feature_sample_ratio < 1.0) {
            int sample_features = std::max(1, static_cast<int>(n_features * feature_sample_ratio));
            std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
            feature_indices.resize(sample_features);
        }
        
        // Build tree
        Node* root = nullptr;
        if (binning_method != NONE) {
            root = buildTreeRecursiveBinned(X, gradients, hessians, all_indices, 
                                           feature_indices, 0, 
                                           sum_gradients, sum_hessians);
        } else {
            root = buildTreeRecursive(X, gradients, hessians, all_indices, 
                                     feature_indices, 0, 
                                     sum_gradients, sum_hessians);
        }
        
        if (!root) {
            continue;
        }
        
        // Create new tree
        Tree new_tree;
        new_tree.root = root;

        // Compute tree predictions
        std::vector<double> new_tree_pred(n_samples);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; ++i) {
            Node* node = root;
            while (node && !node->is_leaf) {
                if (std::isnan(X[i][node->split_feature])) {
                    // Default direction for missing values
                    node = node->left;
                } else {
                    node = (X[i][node->split_feature] <= node->split_value) ? 
                            node->left : node->right;
                }
            }
            new_tree_pred[i] = node ? node->leaf_value : 0.0;
        }

        // Add tree to ensemble
        trees.push_back(new_tree);
        double new_weight = learning_rate;
        tree_weights.push_back(new_weight);

        // DART: adjust weights
        if (useDart && !drop_indices.empty()) {
            // Calculate sum of weights
            double sum_weight = 0.0;
            for (double w : tree_weights) {
                sum_weight += w;
            }
            
            // Numerical stability check
            double old_sum = sum_weight - tree_weights.back();
            // Ensure factor is well-behaved
            double factor = 1.0;
            if (old_sum > 1e-6) {
                factor = old_sum / sum_weight;
            }
            
            // Rescale weights
            for (double &w : tree_weights) {
                w *= factor;
            }
            
            // Update predictions
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_samples; ++i) {
                y_pred_train[i] = factor * y_pred_train[i] + tree_weights.back() * new_tree_pred[i];
            }
        } else {
            // Standard GBM update
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_samples; ++i) {
                y_pred_train[i] += new_weight * new_tree_pred[i];
            }
        }
        
        // Early stopping check
        if (use_early_stopping) {
            // Update validation predictions
            const int n_val = X_val->size();
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_val; ++i) {
                Node* node = root;
                while (node && !node->is_leaf) {
                    if (std::isnan((*X_val)[i][node->split_feature])) {
                        node = node->left;
                    } else {
                        node = ((*X_val)[i][node->split_feature] <= node->split_value) ? 
                                node->left : node->right;
                    }
                }
                
                if (node) {
                    if (useDart && !drop_indices.empty()) {
                        double factor = 1.0;
                        if (tree_weights.size() > 1) {
                            double sum_weight = 0.0;
                            for (double w : tree_weights) {
                                sum_weight += w;
                            }
                            double old_sum = sum_weight - tree_weights.back();
                            if (old_sum > 1e-6) {
                                factor = old_sum / sum_weight;
                            }
                        }
                        y_pred_val[i] = factor * y_pred_val[i] + tree_weights.back() * node->leaf_value;
                    } else {
                        y_pred_val[i] += tree_weights.back() * node->leaf_value;
                    }
                }
            }
            
            // Calculate validation MSE
            double val_mse = 0.0;
            #pragma omp parallel for reduction(+:val_mse) schedule(static)
            for (int i = 0; i < n_val; ++i) {
                double diff = y_pred_val[i] - (*y_val)[i];
                val_mse += diff * diff;
            }
            val_mse /= n_val;
            
            // Check if performance improved
            if (val_mse < best_val_score) {
                best_val_score = val_mse;
                best_iteration = iter;
                // Reset no improvement counter
                std::fill(no_improvement_rounds.begin(), no_improvement_rounds.end(), 0);
            } else {
                // Shift records and add a 1 (indicating no improvement)
                std::rotate(no_improvement_rounds.begin(), 
                           no_improvement_rounds.begin() + 1, 
                           no_improvement_rounds.end());
                no_improvement_rounds.back() = 1;
                
                // If all recent rounds show no improvement, stop early
                if (std::accumulate(no_improvement_rounds.begin(), 
                                   no_improvement_rounds.end(), 0) == early_stopping_rounds) {
                    std::cout << "Early stopping at iteration " << iter 
                              << ". Best iteration: " << best_iteration 
                              << " with validation MSE: " << best_val_score << std::endl;
                    break;
                }
            }
        }
        
        // Reset memory pool for next iteration
        memory_pool.reset();
    }
}

ImprovedGBDT::Node* ImprovedGBDT::buildTreeRecursiveBinned(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& indices, 
    const std::vector<int>& feature_indices,
    int depth,
    double sum_gradients,
    double sum_hessians,
    std::shared_ptr<const std::vector<std::vector<HistogramEntry>>> parent_hist) {
    
    const int n_samples = indices.size();
    
    // Handle empty node
    if (indices.empty()) {
        return nullptr;
    }
    
    // Check stopping conditions
    if (depth >= max_depth || n_samples <= min_samples_leaf) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    
    if (sum_hessians < 1e-10) {  // Only check Hessian to avoid division by zero
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = 0.0;  // Safe default when Hessian is zero
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    // Prepare histograms
    std::vector<std::vector<HistogramEntry>> hist(
        n_features, std::vector<HistogramEntry>(num_bins + 1));
    
    // Use histogram subtraction optimization if appropriate
    int parent_samples = parent_hist && (*parent_hist)[0].size() == num_bins + 1 ? 
                         (*parent_hist)[0][num_bins].count : 0;
    
    bool use_subtraction = parent_hist && parent_samples > 0 && 
                          n_samples > 0 && n_samples < parent_samples / 2;
                          
    if (use_subtraction) {
        // Use smaller node to compute histogram and subtract from parent
        const std::vector<int>* smaller_indices = &indices;
        std::vector<std::vector<HistogramEntry>> smaller_hist(
            n_features, std::vector<HistogramEntry>(num_bins + 1));
        
        buildHistogram(X, *smaller_indices, gradients, hessians, smaller_hist, feature_indices);
        
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t fi = 0; fi < feature_indices.size(); ++fi) {
            for (int b = 0; b <= num_bins; ++b) {
                int f = feature_indices[fi];
                hist[f][b].grad_sum = (*parent_hist)[f][b].grad_sum - smaller_hist[f][b].grad_sum;
                hist[f][b].hess_sum = (*parent_hist)[f][b].hess_sum - smaller_hist[f][b].hess_sum;
                hist[f][b].count = (*parent_hist)[f][b].count - smaller_hist[f][b].count;
            }
        }
    } else {
        // Compute histogram directly
        buildHistogram(X, indices, gradients, hessians, hist, feature_indices);
    }
    
    // Find best split
    int best_feature = -1;
    int best_bin = 0;
    double best_gain = 0.0;
    
    bool found_split = findBestSplit(hist, sum_gradients, sum_hessians, 
        best_feature, best_bin, best_gain, feature_indices);
    
    // If no good split found, make a leaf
    if (!found_split) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    
    // Get split value from bin
    double best_split_value = getSplitValueFromBin(best_feature, best_bin);
    
    // Split the node
    std::vector<int>& left_indices = memory_pool.get_vector();
    std::vector<int>& right_indices = memory_pool.get_vector();
    
    double left_grad_sum = 0.0;
    double left_hess_sum = 0.0;
    splitNodeHistogram(X, indices, best_feature, best_bin, best_split_value,
        left_indices, right_indices, left_grad_sum, left_hess_sum,
        gradients, hessians);
    double right_grad_sum = sum_gradients - left_grad_sum;
    double right_hess_sum = sum_hessians - left_hess_sum;
    
    // Create the split node
    Node* node = new Node();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_value = best_split_value;
    node->sum_grad = sum_gradients;
    node->sum_hess = sum_hessians;
    node->sample_count = n_samples;

    auto hist_ptr = std::make_shared<std::vector<std::vector<HistogramEntry>>>(std::move(hist));
    
    // Build child nodes
    node->left = buildTreeRecursiveBinned(X, gradients, hessians, left_indices, 
                                        feature_indices, depth + 1, 
                                        left_grad_sum, left_hess_sum, 
                                        hist_ptr);
    
    node->right = buildTreeRecursiveBinned(X, gradients, hessians, right_indices, 
                                         feature_indices, depth + 1, 
                                         right_grad_sum, right_hess_sum, 
                                         hist_ptr);
    
    // If both children are nullptr, convert to leaf
    if (!node->left && !node->right) {
        node->is_leaf = true;
        node->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
    }
    
    return node;
}

std::vector<double> ImprovedGBDT::predict(const std::vector<std::vector<double>>& X) const {
    const int n_samples = X.size();
    std::vector<double> predictions(n_samples, initial_prediction);
    
    // Use better threshold for parallelization
    const int parallel_threshold = 
        std::min(1000, std::max(32, static_cast<int>(trees.size() * 2)));
    
    if (n_samples < parallel_threshold) {
        // Direct prediction for small datasets
        for (int i = 0; i < n_samples; ++i) {
            predictions[i] = predict(X[i]);
        }
    } else {
        // Parallel prediction for larger datasets
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < n_samples; ++i) {
                const auto& x = X[i];
                double pred = initial_prediction;
                
                for (size_t j = 0; j < trees.size(); ++j) {
                    const Node* node = trees[j].root;
                    const double weight = tree_weights[j];
                    
                    while (node && !node->is_leaf) {
                        if (std::isnan(x[node->split_feature])) {
                            // Handle missing values
                            node = node->left;
                        } else {
                            node = (x[node->split_feature] <= node->split_value) ? 
                                   node->left : node->right;
                        }
                    }
                    
                    if (node) {
                        pred += weight * node->leaf_value;
                    }
                }
                
                predictions[i] = pred;
            }
        }
    }
    
    return predictions;
}

double ImprovedGBDT::predict(const std::vector<double>& x) const {
    double pred = initial_prediction;
    
    for (size_t j = 0; j < trees.size(); ++j) {
        const Node* node = trees[j].root;
        const double weight = tree_weights[j];
        
        // Navigate down the tree
        while (node && !node->is_leaf) {
            if (std::isnan(x[node->split_feature])) {
                // Handle missing values
                node = node->left;
            } else {
                node = (x[node->split_feature] <= node->split_value) ? 
                       node->left : node->right;
            }
        }
        
        if (node) {
            pred += weight * node->leaf_value;
        }
    }
    
    return pred;
}

ImprovedGBDT::Node* ImprovedGBDT::buildTreeRecursive(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& indices, 
    const std::vector<int>& feature_indices,
    int depth,
    double sum_gradients,
    double sum_hessians) {
    
    const int n_samples = indices.size();
    
    // Handle empty node
    if (indices.empty()) {
        return nullptr;
    }

    // Check stopping conditions
    if (depth >= max_depth || n_samples <= min_samples_leaf) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    
    // Check for near-zero gradient or hessian
    if (sum_hessians < 1e-10) {  // Only check Hessian to avoid division by zero
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = 0.0;  // Safe default when Hessian is zero
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    // Find best split
    double best_gain = 0.0;
    int best_feature = -1;
    double best_split_value = 0.0;
    
    // Sort and evaluate splits for each feature
    #pragma omp parallel
    {
        double thread_best_gain = 0.0;
        int thread_best_feature = -1;
        double thread_best_split = 0.0;
        std::vector<std::pair<double, int>> sort_buffer(n_samples);
        
        #pragma omp for schedule(dynamic)
        for (size_t fi = 0; fi < feature_indices.size(); ++fi) {
            int f = feature_indices[fi];
            
            // Collect feature values and indices
            for (int i = 0; i < n_samples; ++i) {
                int idx = indices[i];
                sort_buffer[i] = {X[idx][f], idx};
            }
            
            // Sort by feature value
            std::sort(sort_buffer.begin(), sort_buffer.end());
            
            // Tracking variables for finding best split
            double left_grad = 0.0;
            double left_hess = 0.0;
            int left_count = 0;
            
            // Consider each potential split point
            for (int i = 0; i < n_samples - 1; ++i) {
                int idx = sort_buffer[i].second;
                double value = sort_buffer[i].first;
                
                if (std::isnan(value)) continue;  // Skip NaNs
                
                // Update left partition
                left_grad += gradients[idx];
                left_hess += hessians[idx];
                left_count++;
                
                // Skip duplicate values
                if (i < n_samples - 1 && value == sort_buffer[i + 1].first) {
                    continue;
                }
                
                // Skip if doesn't meet min_samples_leaf
                if (left_count < min_samples_leaf || 
                    (n_samples - left_count) < min_samples_leaf) {
                    continue;
                }
                
                // Calculate right partition values
                double right_grad = sum_gradients - left_grad;
                double right_hess = sum_hessians - left_hess;
                
                // Calculate gain
                double gain = calculateSplitGain(
                    left_grad, left_hess, 
                    right_grad, right_hess, 
                    sum_gradients, sum_hessians
                );
                
                // Update best split if better
                if (gain > thread_best_gain) {
                    thread_best_gain = gain;
                    thread_best_feature = f;
                    
                    // Set split point between this value and next value
                    if (i < n_samples - 1) {
                        thread_best_split = (value + sort_buffer[i + 1].first) / 2.0;
                    } else {
                        thread_best_split = value;
                    }
                }
            }
        }
        
        // Combine results from all threads
        #pragma omp critical
        {
            if (thread_best_gain > best_gain) {
                best_gain = thread_best_gain;
                best_feature = thread_best_feature;
                best_split_value = thread_best_split;
            }
        }
    }
    
    // If no good split found, make a leaf
    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    
    // Split the node
    std::vector<int>& left_indices  = memory_pool.get_vector();
    std::vector<int>& right_indices = memory_pool.get_vector();
    left_indices.reserve(n_samples/2);
    right_indices.reserve(n_samples/2);
    
    double left_grad_sum = 0.0;
    double left_hess_sum = 0.0;
    
    // Perform the split
    if (n_samples > MIN_SAMPLES_FOR_PARALLEL) {
        // Parallel split for larger nodes
        struct ThreadLocalSplit {
            std::vector<int> left;
            std::vector<int> right;
            double grad_sum = 0.0;
            double hess_sum = 0.0;
        };
        
        std::vector<ThreadLocalSplit> thread_splits(omp_get_max_threads());
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            auto& local_split = thread_splits[thread_id];
            local_split.left.reserve(n_samples / omp_get_max_threads() / 2);
            local_split.right.reserve(n_samples / omp_get_max_threads() / 2);
            
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n_samples; ++i) {
                int idx = indices[i];
                if (std::isnan(X[idx][best_feature]) || X[idx][best_feature] <= best_split_value) {
                    local_split.left.push_back(idx);
                    local_split.grad_sum += gradients[idx];
                    local_split.hess_sum += hessians[idx];
                } else {
                    local_split.right.push_back(idx);
                }
            }
        }
        
        // Combine results from all threads
        for (auto& split : thread_splits) {
            left_indices.insert(left_indices.end(), split.left.begin(), split.left.end());
            right_indices.insert(right_indices.end(), split.right.begin(), split.right.end());
            left_grad_sum += split.grad_sum;
            left_hess_sum += split.hess_sum;
        }
    } else {
        // Serial split for smaller nodes
        for (int idx : indices) {
            if (std::isnan(X[idx][best_feature]) || X[idx][best_feature] <= best_split_value) {
                left_indices.push_back(idx);
                left_grad_sum += gradients[idx];
                left_hess_sum += hessians[idx];
            } else {
                right_indices.push_back(idx);
            }
        }
    }
    
    double right_grad_sum = sum_gradients - left_grad_sum;
    double right_hess_sum = sum_hessians - left_hess_sum;
    
    // Create the split node
    Node* node = new Node();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_value = best_split_value;
    node->sum_grad = sum_gradients;
    node->sum_hess = sum_hessians;
    node->sample_count = n_samples;
    
    // Build child nodes
    node->left = buildTreeRecursive(X, gradients, hessians, left_indices, 
                                  feature_indices, depth + 1, 
                                  left_grad_sum, left_hess_sum);
    
    node->right = buildTreeRecursive(X, gradients, hessians, right_indices, 
                                   feature_indices, depth + 1, 
                                   right_grad_sum, right_hess_sum);
    
    // If both children are nullptr, convert to leaf
    if (!node->left && !node->right) {
        node->is_leaf = true;
        node->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
    }
    
    return node;
}

std::vector<double> ImprovedGBDT::featureImportance() const {
    // Initialize feature importance vector
    std::vector<double> importance(n_features, 0.0);
    
    // Total gain accumulator
    double total_gain = 0.0;
    
    // Function to recursively calculate importance
    std::function<void(const Node*, double, std::vector<double>&, double&)> processNode = 
        [&](const Node* node, double parent_weight, std::vector<double>& imp, double& total) {
            if (!node || node->is_leaf) return;
            
            // Calculate gain for this node - compute directly to avoid issues
            double gain = 0.0;
            
            // Left child contribution
            double left_score = 0.0;
            if (node->left && node->left->sum_hess > 0) {
                left_score = node->left->sum_grad * node->left->sum_grad / 
                           (node->left->sum_hess + l2_reg);
            }
            
            // Right child contribution
            double right_score = 0.0;
            if (node->right && node->right->sum_hess > 0) {
                right_score = node->right->sum_grad * node->right->sum_grad / 
                            (node->right->sum_hess + l2_reg);
            }
            
            // Parent node score
            double parent_score = 0.0;
            if (node->sum_hess > 0) {
                parent_score = node->sum_grad * node->sum_grad / (node->sum_hess + l2_reg);
            }
            
            // Calculate split gain as improvement
            gain = left_score + right_score - parent_score;
            
            // Apply tree weight
            gain *= parent_weight;
            
            // Ensure non-negative gain
            gain = std::max(0.0, gain);
            
            // Accumulate total gain
            total += gain;
            
            // Add to feature importance
            if (node->split_feature >= 0 && node->split_feature < n_features) {
                imp[node->split_feature] += gain;
            }
            
            // Process child nodes
            processNode(node->left, parent_weight, imp, total);
            processNode(node->right, parent_weight, imp, total);
        };
    
    // Process all trees with their weights
    for (size_t i = 0; i < trees.size(); ++i) {
        processNode(trees[i].root, tree_weights[i], importance, total_gain);
    }
    
    // If no gain found, use feature frequency instead
    if (total_gain <= 0.0) {
        std::fill(importance.begin(), importance.end(), 0.0);
        total_gain = 0.0;
        
        // Count feature frequency in splits
        std::function<void(const Node*)> countFeatures = 
            [&](const Node* node) {
                if (!node || node->is_leaf) return;
                
                if (node->split_feature >= 0 && node->split_feature < n_features) {
                    importance[node->split_feature] += 1.0;
                    total_gain += 1.0;
                }
                
                countFeatures(node->left);
                countFeatures(node->right);
            };
        
        for (const auto& tree : trees) {
            countFeatures(tree.root);
        }
    }
    
    // Normalize
    if (total_gain > 0.0) {
        for (double& imp : importance) {
            imp /= total_gain;
        }
    }
    
    return importance;
}

// Save model to file
bool ImprovedGBDT::saveModel(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    
    // Write header information
    out << "ImprovedGBDT_MODEL_v1.0" << std::endl;
    
    // Write model parameters
    out << n_estimators << " " 
        << max_depth << " " 
        << learning_rate << " "
        << (useDart ? 1 : 0) << " " 
        << drop_rate << " " 
        << skip_rate << " "
        << static_cast<int>(binning_method) << " " 
        << num_bins << " "
        << min_samples_leaf << " " 
        << l2_reg << " "
        << feature_sample_ratio << " " 
        << early_stopping_rounds << " "
        << n_features << " "
        << initial_prediction << " "
        << static_cast<int>(trees.size()) << std::endl;
    
    // Write tree weights
    for (double weight : tree_weights) {
        out << weight << " ";
    }
    out << std::endl;
    
    // Write trees
    for (const auto& tree : trees) {
        serializeNode(tree.root, out);
    }
    
    out.close();
    return true;
}

// Load model from file
bool ImprovedGBDT::loadModel(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    
    // Clear existing trees
    for (auto& tree : trees) {
        freeTree(tree.root);
    }
    trees.clear();
    tree_weights.clear();
    
    // Read header
    std::string header;
    std::getline(in, header);
    if (header != "ImprovedGBDT_MODEL_v1.0") {
        return false;
    }
    
    // Read model parameters
    int dart_flag, bin_method, num_trees;
    in >> n_estimators 
       >> max_depth 
       >> learning_rate
       >> dart_flag 
       >> drop_rate 
       >> skip_rate
       >> bin_method 
       >> num_bins
       >> min_samples_leaf 
       >> l2_reg
       >> feature_sample_ratio 
       >> early_stopping_rounds
       >> n_features
       >> initial_prediction
       >> num_trees;
    
    useDart = (dart_flag != 0);
    binning_method = static_cast<BinningMethod>(bin_method);
    
    // Read tree weights
    tree_weights.resize(num_trees);
    for (int i = 0; i < num_trees; ++i) {
        in >> tree_weights[i];
    }
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    // Read trees
    trees.resize(num_trees);
    for (int i = 0; i < num_trees; ++i) {
        trees[i].root = deserializeNode(in);
        if (!trees[i].root) {
            return false;
        }
    }
    
    in.close();
    return true;
}

// Serialize a node to output stream
void ImprovedGBDT::serializeNode(const Node* node, std::ostream& out) const {
    if (!node) {
        out << "NULL" << std::endl;
        return;
    }
    
    if (node->is_leaf) {
        // Leaf node format: LEAF value sample_count grad hess
        out << "LEAF " 
            << node->leaf_value << " " 
            << node->sample_count << " " 
            << node->sum_grad << " " 
            << node->sum_hess << std::endl;
    } else {
        // Internal node format: NODE feature value sample_count grad hess
        out << "NODE " 
            << node->split_feature << " " 
            << node->split_value << " " 
            << node->sample_count << " " 
            << node->sum_grad << " " 
            << node->sum_hess << std::endl;
        
        // Recursively serialize children
        serializeNode(node->left, out);
        serializeNode(node->right, out);
    }
}

// Deserialize a node from input stream
ImprovedGBDT::Node* ImprovedGBDT::deserializeNode(std::istream& in) {
    std::string node_type;
    in >> node_type;
    
    if (node_type == "NULL") {
        return nullptr;
    }
    
    Node* node = new Node();
    
    if (node_type == "LEAF") {
        node->is_leaf = true;
        in >> node->leaf_value >> node->sample_count >> node->sum_grad >> node->sum_hess;
    } else if (node_type == "NODE") {
        node->is_leaf = false;
        in >> node->split_feature >> node->split_value >> node->sample_count 
           >> node->sum_grad >> node->sum_hess;
        
        // Recursively deserialize children
        node->left = deserializeNode(in);
        node->right = deserializeNode(in);
    } else {
        delete node;
        return nullptr;
    }
    
    return node;
}

// Get training parameters as map
std::map<std::string, std::string> ImprovedGBDT::getTrainingParameters() const {
    std::map<std::string, std::string> params;
    params["NumEstimators"] = std::to_string(n_estimators);
    params["MaxDepth"] = std::to_string(max_depth);
    params["LearningRate"] = std::to_string(learning_rate);
    params["UseDart"] = useDart ? "1" : "0";
    params["DropRate"] = std::to_string(drop_rate);
    params["SkipRate"] = std::to_string(skip_rate);
    params["BinningMethod"] = std::to_string(static_cast<int>(binning_method));
    params["NumBins"] = std::to_string(num_bins);
    params["MinSamplesLeaf"] = std::to_string(min_samples_leaf);
    params["L2Reg"] = std::to_string(l2_reg);
    params["FeatureSampleRatio"] = std::to_string(feature_sample_ratio);
    params["EarlyStoppingRounds"] = std::to_string(early_stopping_rounds);
    params["InitialPrediction"] = std::to_string(initial_prediction);
    return params;
}

// Get training parameters as string
std::string ImprovedGBDT::getTrainingParametersString() const {
    std::ostringstream oss;
    oss << "Training Parameters:" << std::endl;
    oss << "  - Number of Estimators: " << n_estimators << std::endl;
    oss << "  - Max Depth: " << max_depth << std::endl;
    oss << "  - Learning Rate: " << learning_rate << std::endl;
    oss << "  - Use DART: " << (useDart ? "Yes" : "No") << std::endl;
    
    if (useDart) {
        oss << "  - Drop Rate: " << drop_rate << std::endl;
        oss << "  - Skip Rate: " << skip_rate << std::endl;
    }
    
    oss << "  - Binning Method: ";
    switch (binning_method) {
        case NONE: oss << "None"; break;
        case QUANTILE: oss << "Quantile"; break;
        case FREQUENCY: oss << "Frequency"; break;
    }
    oss << std::endl;
    
    if (binning_method != NONE) {
        oss << "  - Number of Bins: " << num_bins << std::endl;
    }
    
    oss << "  - Min Samples in Leaf: " << min_samples_leaf << std::endl;
    oss << "  - L2 Regularization: " << l2_reg << std::endl;
    oss << "  - Feature Sampling Ratio: " << feature_sample_ratio << std::endl;
    oss << "  - Early Stopping Rounds: " << early_stopping_rounds << std::endl;
    oss << "  - Initial Prediction: " << initial_prediction << std::endl;
    
    return oss.str();
}