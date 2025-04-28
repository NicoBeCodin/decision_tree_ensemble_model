#ifndef BINNING_METHODS_H
#define BINNING_METHODS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <limits>

namespace Binning {

class QuantileSketch {
public:

    QuantileSketch(int num_bins = 256) : num_bins(num_bins) {}


    void build(const std::vector<std::vector<double>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        int n_features = X[0].size();
        feature_cuts.resize(n_features);
        

        for (int feature_idx = 0; feature_idx < n_features; ++feature_idx) {
            std::vector<double> values;
            values.reserve(X.size());
            
  
            for (const auto& sample : X) {
                double val = sample[feature_idx];
                if (!std::isnan(val)) {
                    values.push_back(val);
                }
            }
            
            if (values.empty()) {
     
                feature_cuts[feature_idx] = {0.0};
                continue;
            }
    
            std::sort(values.begin(), values.end());
            
  
            auto last = std::unique(values.begin(), values.end());
            values.erase(last, values.end());
     
            if (values.size() <= static_cast<size_t>(num_bins)) {
                feature_cuts[feature_idx] = values;
                continue;
            }

            std::vector<double> cuts;
            cuts.reserve(num_bins);
            
            for (int i = 0; i < num_bins; ++i) {
                double pos = i * (values.size() - 1.0) / num_bins;
                int idx = static_cast<int>(pos);
                double alpha = pos - idx;
                
                if (alpha < 1e-10) {
                    cuts.push_back(values[idx]);
                } else {
              
                    cuts.push_back(values[idx] * (1.0 - alpha) + values[idx + 1] * alpha);
                }
            }
            
            if (!cuts.empty() && cuts.front() > values.front()) {
                cuts.front() = values.front();
            }
            if (!cuts.empty() && cuts.back() < values.back()) {
                cuts.back() = values.back();
            }
            
            feature_cuts[feature_idx] = cuts;
        }
    }

    int getBin(double value, int feature_idx) const {
        if (feature_idx < 0 || feature_idx >= static_cast<int>(feature_cuts.size())) {
            return 0; 
        }
        
        const std::vector<double>& cuts = feature_cuts[feature_idx];
        if (cuts.empty()) {
            return 0;
        }
        
      
        if (std::isnan(value)) return 0;
        if (value <= cuts.front()) return 0;
        if (value > cuts.back()) return cuts.size();
        
    
        auto it = std::upper_bound(cuts.begin(), cuts.end(), value);
        return std::distance(cuts.begin(), it);
    }

 
    double getSplitValue(int feature_idx, int bin_idx) const {
        if (feature_idx < 0 || feature_idx >= static_cast<int>(feature_cuts.size())) {
            return 0.0;
        }
        
        const std::vector<double>& cuts = feature_cuts[feature_idx];
        if (bin_idx < 0 || bin_idx >= static_cast<int>(cuts.size())) {
            return 0.0;
        }
        
        return cuts[bin_idx];
    }

private:
    int num_bins;
    std::vector<std::vector<double>> feature_cuts; 
};


class FrequencyBinning {
public:
   
    FrequencyBinning(int num_bins = 256) : num_bins(num_bins) {}
    
  
    void build(const std::vector<std::vector<double>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        int n_features = X[0].size();
        feature_cuts.resize(n_features);
        
    
        for (int feature_idx = 0; feature_idx < n_features; ++feature_idx) {
            std::vector<double> values;
            values.reserve(X.size());
            
          
            for (const auto& sample : X) {
                double val = sample[feature_idx];
                if (!std::isnan(val)) {
                    values.push_back(val);
                }
            }
            
            if (values.empty()) {
               
                feature_cuts[feature_idx] = {0.0};
                continue;
            }
       
            std::sort(values.begin(), values.end());
            
        
            auto last = std::unique(values.begin(), values.end());
            values.erase(last, values.end());
            
            if (values.size() <= static_cast<size_t>(num_bins)) {
                feature_cuts[feature_idx] = values;
                continue;
            }
            
       
            std::vector<double> cuts;
            cuts.reserve(num_bins - 1); 
            
        
            int samples_per_bin = values.size() / num_bins;
            int remainder = values.size() % num_bins;
            
            size_t current_pos = 0;
            for (int bin = 0; bin < num_bins - 1; ++bin) {
            
                int current_bin_size = samples_per_bin + (bin < remainder ? 1 : 0);
                current_pos += current_bin_size;
                
                if (current_pos < values.size()) {
                
                    size_t next_pos = current_pos;
                    while (next_pos < values.size() - 1 && 
                           std::abs(values[next_pos] - values[current_pos]) < 1e-10) {
                        next_pos++;
                    }
                    
              
                    if (next_pos < values.size() - 1) {
                        cuts.push_back((values[current_pos] + values[next_pos]) / 2.0);
                    } else {
                        cuts.push_back(values[current_pos]);
                    }
                }
            }
            
            feature_cuts[feature_idx] = cuts;
        }
    }
    

    int getBin(double value, int feature_idx) const {
        if (feature_idx < 0 || feature_idx >= static_cast<int>(feature_cuts.size())) {
            return 0;
        }
        
        const std::vector<double>& cuts = feature_cuts[feature_idx];
        if (cuts.empty()) {
            return 0;
        }
        
    
        if (std::isnan(value)) return 0;
        if (value <= cuts.front()) return 0;
        if (value > cuts.back()) return cuts.size();
        
      
        auto it = std::upper_bound(cuts.begin(), cuts.end(), value);
        return std::distance(cuts.begin(), it);
    }
    
  
    double getSplitValue(int feature_idx, int bin_idx) const {
        if (feature_idx < 0 || feature_idx >= static_cast<int>(feature_cuts.size())) {
            return 0.0;
        }
        
        const std::vector<double>& cuts = feature_cuts[feature_idx];
        if (bin_idx < 0 || bin_idx >= static_cast<int>(cuts.size())) {
            return 0.0;
        }
        
        return cuts[bin_idx];
    }

private:
    int num_bins;
    std::vector<std::vector<double>> feature_cuts; 
};

} 

#endif 