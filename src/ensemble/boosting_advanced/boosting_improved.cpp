#include "boosting_improved.h"
#include <omp.h>
#include <memory>
#include <algorithm>
#include <numeric>


void ImprovedGBDT::precomputeFeatureBins(const std::vector<std::vector<double>>& X) {
    if (binning_method == NONE || X.empty()) return;
    
    int n_samples = X.size();
    int n_features = X[0].size();
    

    bin_cache.feature_bins.resize(n_features);
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int f = 0; f < n_features; ++f) {
        bin_cache.feature_bins[f].resize(n_samples);
        for (int i = 0; i < n_samples; ++i) {
         
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
    std::vector<std::vector<HistogramEntry>>& hist) {
    
    int n_features = X[0].size();
    

    for (auto& feature_hist : hist) {
        std::fill(feature_hist.begin(), feature_hist.end(), HistogramEntry());
    }
    
    if (indices.size() < MIN_SAMPLES_FOR_PARALLEL) {
     
        for (int idx : indices) {
            for (int f = 0; f < n_features; ++f) {
                int bin = getBinIndex(X[idx][f], f, idx);
                hist[f][bin].grad_sum += gradients[idx];
                hist[f][bin].hess_sum += hessians[idx];
                hist[f][bin].count++;
            }
        }
    } else {
  
        const int max_threads = omp_get_max_threads();
        std::vector<std::vector<std::vector<HistogramEntry>>> thread_histograms(
            max_threads, std::vector<std::vector<HistogramEntry>>(
                n_features, std::vector<HistogramEntry>(num_bins + 1)
            )
        );
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            auto& local_hist = thread_histograms[thread_id];
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                for (int f = 0; f < n_features; ++f) {
                    int bin = getBinIndex(X[idx][f], f, idx);
                    local_hist[f][bin].grad_sum += gradients[idx];
                    local_hist[f][bin].hess_sum += hessians[idx];
                    local_hist[f][bin].count++;
                }
            }
        }
        
      
        #pragma omp parallel for collapse(2)
        for (int f = 0; f < n_features; ++f) {
            for (int bin = 0; bin <= num_bins; ++bin) {
                for (int t = 0; t < max_threads; ++t) {
                    hist[f][bin].grad_sum += thread_histograms[t][f][bin].grad_sum;
                    hist[f][bin].hess_sum += thread_histograms[t][f][bin].hess_sum;
                    hist[f][bin].count += thread_histograms[t][f][bin].count;
                }
            }
        }
    }
}


bool ImprovedGBDT::findBestSplit(
    const std::vector<std::vector<HistogramEntry>>& hist,
    double sum_gradients,
    double sum_hessians,
    int& best_feature,
    int& best_bin,
    double& best_gain) {
    
    int n_features = hist.size();
    best_feature = -1;
    best_bin = 0;
    best_gain = 0.0;
    
    #pragma omp parallel
    {
        double local_best_gain = 0.0;
        int local_best_feature = -1;
        int local_best_bin = 0;
        
        #pragma omp for schedule(dynamic, 1)
        for (int f = 0; f < n_features; ++f) {
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
                
                if (left_count < 1 || right_count < 1) continue;
                
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
                if (X[idx][best_feature] <= best_split_value) {
                    left_count++;
                } else {
                    right_count++;
                }
            }
            
            left_counts[thread_id + 1] = left_count;
            right_counts[thread_id + 1] = right_count;
        }
        
    
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
                if (X[idx][best_feature] <= best_split_value) {
                    left_buffer[left_offset++] = idx;
                    left_grad_sum += gradients[idx];
                    left_hess_sum += hessians[idx];
                } else {
                    right_buffer[right_offset++] = idx;
                }
            }
        }
        
   
        left_indices.assign(left_buffer.begin(), left_buffer.begin() + left_counts[omp_get_max_threads()]);
        right_indices.assign(right_buffer.begin(), right_buffer.begin() + right_counts[omp_get_max_threads()]);
    } else {
      
        for (int idx : indices) {
            if (X[idx][best_feature] <= best_split_value) {
                left_indices.push_back(idx);
                left_grad_sum += gradients[idx];
                left_hess_sum += hessians[idx];
            } else {
                right_indices.push_back(idx);
            }
        }
    }
}



void ImprovedGBDT::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    int n_samples = X.size();
    if (n_samples == 0) return;
    int n_features = X[0].size();
    

    num_threads = omp_get_max_threads();
    
 
    memory_pool.init(num_threads * 4, n_samples);

  
    if (binning_method == QUANTILE) {
        quantile_binner = std::make_unique<Binning::QuantileSketch>(num_bins);
        quantile_binner->build(X);
    } else if (binning_method == FREQUENCY) {
        frequency_binner = std::make_unique<Binning::FrequencyBinning>(num_bins);
        frequency_binner->build(X);
    }
    
 
    if (binning_method != NONE) {
        precomputeFeatureBins(X);
    }

 
    double sum_y = 0.0;
    #pragma omp parallel for reduction(+:sum_y) if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
    for (int i = 0; i < n_samples; ++i) {
        sum_y += y[i];
    }
    initial_prediction = sum_y / n_samples;
    

    y_pred_train.assign(n_samples, initial_prediction);

 
    trees.reserve(n_estimators);
    tree_weights.reserve(n_estimators);

  
    std::uniform_real_distribution<double> dist(0.0, 1.0);


    std::vector<double> gradients(n_samples);
    std::vector<double> hessians(n_samples, 1.0); 

 
    for (int iter = 0; iter < n_estimators; ++iter) {
    
        if (useDart && skip_rate > 0.0) {
            double skip_sample = dist(rng);
            if (skip_sample < skip_rate) {
                continue;
            }
        }

      
        std::vector<int> drop_indices;
        std::vector<double> pred_for_grad = y_pred_train;
        
        if (useDart && !trees.empty() && drop_rate > 0.0) {
           
            for (int j = 0; j < (int)trees.size(); ++j) {
                double r = dist(rng);
                if (r < drop_rate) {
                    drop_indices.push_back(j);
                }
            }
            
         
            if (drop_indices.size() == trees.size()) {
                drop_indices.pop_back();
            }
            
         
            if (!drop_indices.empty()) {
                #pragma omp parallel for if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
                for (int i = 0; i < n_samples; ++i) {
                    for (int drop_idx : drop_indices) {
                        const Node* node = trees[drop_idx].root;
                        while (node && !node->is_leaf) {
                            node = (X[i][node->split_feature] <= node->split_value) ? 
                                    node->left : node->right;
                        }
                        if (node) {
                            pred_for_grad[i] -= tree_weights[drop_idx] * node->leaf_value;
                        }
                    }
                }
            }
        }

   
        #pragma omp parallel for if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
        for (int i = 0; i < n_samples; ++i) {
            gradients[i] = pred_for_grad[i] - y[i]; 
            hessians[i] = 1.0;                   
        }

     
        std::vector<int> all_indices(n_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
  
        double sum_gradients = 0.0;
        double sum_hessians = static_cast<double>(n_samples); 
        
        #pragma omp parallel for reduction(+:sum_gradients) if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
        for (int i = 0; i < n_samples; ++i) {
            sum_gradients += gradients[i];
        }
        
   
        Node* root = nullptr;
        if (binning_method != NONE) {
            root = buildTreeRecursiveBinned(X, gradients, hessians, all_indices, 0, 
                                          sum_gradients, sum_hessians);
        } else {
            root = buildTreeRecursive(X, gradients, hessians, all_indices, 0, 
                                    sum_gradients, sum_hessians);
        }
        
  
        if (!root) {
            continue;
        }
        
        Tree new_tree;
        new_tree.root = root;

        std::vector<double> new_tree_pred(n_samples);
        
        #pragma omp parallel for if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
        for (int i = 0; i < n_samples; ++i) {
            const Node* node = root;
            while (node && !node->is_leaf) {
                node = (X[i][node->split_feature] <= node->split_value) ? 
                        node->left : node->right;
            }
            new_tree_pred[i] = node ? node->leaf_value : 0.0;
        }

    
        trees.push_back(new_tree);
        double new_weight = learning_rate;
        tree_weights.push_back(new_weight);

     
        if (useDart && !drop_indices.empty()) {
        
            double sum_weight = 0.0;
            for (double w : tree_weights) {
                sum_weight += w;
            }
            double old_sum = sum_weight - tree_weights.back();
            double factor = (old_sum > 0.0 ? old_sum / sum_weight : 1.0);
            
       
            for (double &w : tree_weights) {
                w *= factor;
            }
            
        
            #pragma omp parallel for if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
            for (int i = 0; i < n_samples; ++i) {
                y_pred_train[i] = factor * y_pred_train[i] + tree_weights.back() * new_tree_pred[i];
            }
        } else {
          
            #pragma omp parallel for if(n_samples > MIN_SAMPLES_FOR_PARALLEL)
            for (int i = 0; i < n_samples; ++i) {
                y_pred_train[i] += new_weight * new_tree_pred[i];
            }
        }
        
     
        memory_pool.reset();
    }
}


ImprovedGBDT::Node* ImprovedGBDT::buildTreeRecursiveBinned(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& indices, 
    int depth,
    double sum_gradients,
    double sum_hessians,
    const std::vector<std::vector<HistogramEntry>>* parent_hist) {
    
    const int n_samples = indices.size();
    const int n_features = X[0].size();
    
    if (indices.empty()) {
        return nullptr;
    }
    

    if (depth >= max_depth || n_samples <= 1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    

    if (std::abs(sum_gradients) < 1e-10 || sum_hessians < 1e-10) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }

    std::vector<std::vector<HistogramEntry>> hist(
        n_features, std::vector<HistogramEntry>(num_bins + 1));
 
    if (parent_hist && parent_hist->size() == n_features && 
        (*parent_hist)[0].size() == num_bins + 1 && 
        n_samples > n_samples / 3) { 
        
 
        const std::vector<int>* smaller_indices = &indices;
        std::vector<std::vector<HistogramEntry>> smaller_hist(
            n_features, std::vector<HistogramEntry>(num_bins + 1));
        
  
        buildHistogram(X, *smaller_indices, gradients, hessians, smaller_hist);
        
  
        #pragma omp parallel for collapse(2) if(n_features > 10)
        for (int f = 0; f < n_features; ++f) {
            for (int b = 0; b <= num_bins; ++b) {
                hist[f][b].grad_sum = (*parent_hist)[f][b].grad_sum - smaller_hist[f][b].grad_sum;
                hist[f][b].hess_sum = (*parent_hist)[f][b].hess_sum - smaller_hist[f][b].hess_sum;
                hist[f][b].count = (*parent_hist)[f][b].count - smaller_hist[f][b].count;
            }
        }
    } else {
     
        buildHistogram(X, indices, gradients, hessians, hist);
    }
    

    int best_feature = -1;
    int best_bin = 0;
    double best_gain = 0.0;
    
    bool found_split = findBestSplit(hist, sum_gradients, sum_hessians, 
        best_feature, best_bin, best_gain);
    

    if (!found_split) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    

    double best_split_value = getSplitValueFromBin(best_feature, best_bin);
    
 
    std::vector<int>& left_indices = memory_pool.get_vector();
    std::vector<int>& right_indices = memory_pool.get_vector();
    
 
    double left_grad_sum = 0.0;
    double left_hess_sum = 0.0;
    splitNodeHistogram(X, indices, best_feature, best_bin, best_split_value,
        left_indices, right_indices, left_grad_sum, left_hess_sum,
        gradients, hessians);
    double right_grad_sum = sum_gradients - left_grad_sum;
    double right_hess_sum = sum_hessians - left_hess_sum;
    
 
    Node* node = new Node();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_value = best_split_value;
    node->sum_grad = sum_gradients;
    node->sum_hess = sum_hessians;
    node->sample_count = n_samples;
    

    node->left = buildTreeRecursiveBinned(X, gradients, hessians, left_indices, 
                                        depth + 1, left_grad_sum, left_hess_sum, 
                                        &hist);
    
    node->right = buildTreeRecursiveBinned(X, gradients, hessians, right_indices, 
                                         depth + 1, right_grad_sum, right_hess_sum, 
                                         &hist);
    
  
    if (!node->left && !node->right) {
        node->is_leaf = true;
        node->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
    }
    
    return node;
}


std::vector<double> ImprovedGBDT::predict(const std::vector<std::vector<double>>& X) const {
    const int n_samples = X.size();
    std::vector<double> predictions(n_samples, initial_prediction);
    

    if (n_samples < 64) {
        for (int i = 0; i < n_samples; ++i) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
 
    #pragma omp parallel
    {
     
        #pragma omp for schedule(static)
        for (int i = 0; i < n_samples; ++i) {
            const auto& x = X[i];
            
            for (size_t j = 0; j < trees.size(); ++j) {
                const Node* node = trees[j].root;
                const double weight = tree_weights[j];
                
          
                while (node && !node->is_leaf) {
                    node = (x[node->split_feature] <= node->split_value) ? 
                           node->left : node->right;
                }
                
            
                if (node) {
                    predictions[i] += weight * node->leaf_value;
                }
            }
        }
    }
    
    return predictions;
}


ImprovedGBDT::Node* ImprovedGBDT::buildTreeRecursive(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& indices, 
    int depth,
    double sum_gradients,
    double sum_hessians) {
    
    const int n_samples = indices.size();
    const int n_features = X[0].size();
    

    if (indices.empty()) {
        return nullptr;
    }

    if (depth >= max_depth || n_samples <= 1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    
    if (std::abs(sum_gradients) < 1e-10 || sum_hessians < 1e-10) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }

    double best_gain = 0.0;
    int best_feature = -1;
    double best_split_value = 0.0;
    

    struct SortItem {
        double feature_value;
        double gradient;
        double hessian;
        int original_index;
        
        bool operator<(const SortItem& other) const {
            return feature_value < other.feature_value;
        }
    };
    
    std::vector<SortItem> sort_buffer(n_samples);
    

    #pragma omp parallel
    {

        double thread_best_gain = 0.0;
        int thread_best_feature = -1;
        double thread_best_split = 0.0;
        std::vector<SortItem> local_sort_buffer(n_samples);
        
        #pragma omp for schedule(dynamic, 1)
        for (int f = 0; f < n_features; ++f) {
    
            for (int i = 0; i < n_samples; ++i) {
                int idx = indices[i];
                local_sort_buffer[i] = {
                    X[idx][f],
                    gradients[idx],
                    hessians[idx],
                    idx
                };
            }
            
    
            std::sort(local_sort_buffer.begin(), local_sort_buffer.end());
            
       
            double left_grad = 0.0;
            double left_hess = 0.0;
            
            for (int i = 0; i < n_samples - 1; ++i) {
                left_grad += local_sort_buffer[i].gradient;
                left_hess += local_sort_buffer[i].hessian;
                
           
                if (std::abs(local_sort_buffer[i].feature_value - local_sort_buffer[i+1].feature_value) < 1e-10) {
                    continue;
                }
                
           
                double right_grad = sum_gradients - left_grad;
                double right_hess = sum_hessians - left_hess;
                
           
                if (left_hess < 1.0 || right_hess < 1.0) {
                    continue;
                }
                
                double gain = calculateSplitGain(
                    left_grad, left_hess, 
                    right_grad, right_hess, 
                    sum_gradients, sum_hessians
                );
                
    
                if (gain > thread_best_gain) {
                    thread_best_gain = gain;
                    thread_best_feature = f;
                    thread_best_split = (local_sort_buffer[i].feature_value + 
                                      local_sort_buffer[i+1].feature_value) / 2.0;
                }
            }
        }
        
    
        #pragma omp critical
        {
            if (thread_best_gain > best_gain) {
                best_gain = thread_best_gain;
                best_feature = thread_best_feature;
                best_split_value = thread_best_split;
            }
        }
    }
    

    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
        leaf->sum_grad = sum_gradients;
        leaf->sum_hess = sum_hessians;
        leaf->sample_count = n_samples;
        return leaf;
    }
    
 
    std::vector<int>& left_indices = memory_pool.get_vector();
    std::vector<int>& right_indices = memory_pool.get_vector();
    left_indices.reserve(n_samples/2);
    right_indices.reserve(n_samples/2);
    
 
    double left_grad_sum = 0.0;
    double left_hess_sum = 0.0;
 
    if (n_samples > MIN_SAMPLES_FOR_PARALLEL) {
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
                if (X[idx][best_feature] <= best_split_value) {
                    local_split.left.push_back(idx);
                    local_split.grad_sum += gradients[idx];
                    local_split.hess_sum += hessians[idx];
                } else {
                    local_split.right.push_back(idx);
                }
            }
        }
        
      
        for (auto& split : thread_splits) {
            left_indices.insert(left_indices.end(), split.left.begin(), split.left.end());
            right_indices.insert(right_indices.end(), split.right.begin(), split.right.end());
            left_grad_sum += split.grad_sum;
            left_hess_sum += split.hess_sum;
        }
    } else {
     
        for (int idx : indices) {
            if (X[idx][best_feature] <= best_split_value) {
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
    
   
    Node* node = new Node();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_value = best_split_value;
    node->sum_grad = sum_gradients;
    node->sum_hess = sum_hessians;
    node->sample_count = n_samples;
    

    node->left = buildTreeRecursive(X, gradients, hessians, left_indices, 
                                  depth + 1, left_grad_sum, left_hess_sum);
    
    node->right = buildTreeRecursive(X, gradients, hessians, right_indices, 
                                   depth + 1, right_grad_sum, right_hess_sum);
    
 
    if (!node->left && !node->right) {
        node->is_leaf = true;
        node->leaf_value = calculateLeafValue(sum_gradients, sum_hessians);
    }
    
    return node;
}

double ImprovedGBDT::predict(const std::vector<double>& x) const {
    double pred = initial_prediction;
    
    
    for (size_t j = 0; j < trees.size(); ++j) {
        const Node* node = trees[j].root;
        const double weight = tree_weights[j];
        
        while (node && !node->is_leaf) {
            node = (x[node->split_feature] <= node->split_value) ? 
                   node->left : node->right;
        }
        
      
        if (node) {
            pred += weight * node->leaf_value;
        }
    }
    
    return pred;
}