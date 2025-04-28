#ifndef BINNING_METHODS_H
#define BINNING_METHODS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <limits>

namespace Binning {

// 量化草图（Quantile Sketch）算法 - 受XGBoost启发
class QuantileSketch {
public:
    // 构造函数：指定所需的箱子数量
    QuantileSketch(int num_bins = 256) : num_bins(num_bins) {}

    // 根据训练数据构建分箱映射
    void build(const std::vector<std::vector<double>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        int n_features = X[0].size();
        feature_cuts.resize(n_features);
        
        // 为每个特征构建分位数剪切点
        for (int feature_idx = 0; feature_idx < n_features; ++feature_idx) {
            std::vector<double> values;
            values.reserve(X.size());
            
            // 收集该特征的所有非NaN值
            for (const auto& sample : X) {
                double val = sample[feature_idx];
                if (!std::isnan(val)) {
                    values.push_back(val);
                }
            }
            
            if (values.empty()) {
                // 特征全为NaN，添加一个默认切分点
                feature_cuts[feature_idx] = {0.0};
                continue;
            }
            
            // 排序以便计算分位数
            std::sort(values.begin(), values.end());
            
            // 移除重复值
            auto last = std::unique(values.begin(), values.end());
            values.erase(last, values.end());
            
            // 如果唯一值少于设定的分箱数，直接使用所有唯一值作为切分点
            if (values.size() <= static_cast<size_t>(num_bins)) {
                feature_cuts[feature_idx] = values;
                continue;
            }
            
            // 否则根据分位数选择切分点
            std::vector<double> cuts;
            cuts.reserve(num_bins);
            
            for (int i = 0; i < num_bins; ++i) {
                double pos = i * (values.size() - 1.0) / num_bins;
                int idx = static_cast<int>(pos);
                double alpha = pos - idx;
                
                if (alpha < 1e-10) {
                    cuts.push_back(values[idx]);
                } else {
                    // 线性插值
                    cuts.push_back(values[idx] * (1.0 - alpha) + values[idx + 1] * alpha);
                }
            }
            
            // 确保包含最小值和最大值
            if (!cuts.empty() && cuts.front() > values.front()) {
                cuts.front() = values.front();
            }
            if (!cuts.empty() && cuts.back() < values.back()) {
                cuts.back() = values.back();
            }
            
            feature_cuts[feature_idx] = cuts;
        }
    }
    
    // 将单个特征值映射到对应的箱编号
    int getBin(double value, int feature_idx) const {
        if (feature_idx < 0 || feature_idx >= static_cast<int>(feature_cuts.size())) {
            return 0; // 默认箱
        }
        
        const std::vector<double>& cuts = feature_cuts[feature_idx];
        if (cuts.empty()) {
            return 0;
        }
        
        // 处理特殊情况
        if (std::isnan(value)) return 0;
        if (value <= cuts.front()) return 0;
        if (value > cuts.back()) return cuts.size();
        
        // 二分查找定位箱编号
        auto it = std::upper_bound(cuts.begin(), cuts.end(), value);
        return std::distance(cuts.begin(), it);
    }

    // 获取指定特征和箱的切分值（用于决策树分裂）
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
    std::vector<std::vector<double>> feature_cuts; // 每个特征的切分点
};

// 频率分箱（Frequency Binning）算法 - 受LightGBM启发
class FrequencyBinning {
public:
    // 构造函数：指定所需的箱子数量
    FrequencyBinning(int num_bins = 256) : num_bins(num_bins) {}
    
    // 根据训练数据构建分箱映射
    void build(const std::vector<std::vector<double>>& X) {
        if (X.empty() || X[0].empty()) return;
        
        int n_features = X[0].size();
        feature_cuts.resize(n_features);
        
        // 为每个特征构建均匀频率分箱
        for (int feature_idx = 0; feature_idx < n_features; ++feature_idx) {
            std::vector<double> values;
            values.reserve(X.size());
            
            // 收集该特征的所有非NaN值
            for (const auto& sample : X) {
                double val = sample[feature_idx];
                if (!std::isnan(val)) {
                    values.push_back(val);
                }
            }
            
            if (values.empty()) {
                // 特征全为NaN，添加一个默认切分点
                feature_cuts[feature_idx] = {0.0};
                continue;
            }
            
            // 排序以便等频分箱
            std::sort(values.begin(), values.end());
            
            // 如果唯一值少于设定的分箱数，使用所有值
            auto last = std::unique(values.begin(), values.end());
            values.erase(last, values.end());
            
            if (values.size() <= static_cast<size_t>(num_bins)) {
                feature_cuts[feature_idx] = values;
                continue;
            }
            
            // 根据频率选择切分点，使每个箱内样本数量大致相等
            std::vector<double> cuts;
            cuts.reserve(num_bins - 1); // 需要num_bins-1个切分点得到num_bins个箱
            
            // 每个箱的期望样本数
            int samples_per_bin = values.size() / num_bins;
            int remainder = values.size() % num_bins;
            
            size_t current_pos = 0;
            for (int bin = 0; bin < num_bins - 1; ++bin) {
                // 当前箱的大小，考虑余数分配
                int current_bin_size = samples_per_bin + (bin < remainder ? 1 : 0);
                current_pos += current_bin_size;
                
                if (current_pos < values.size()) {
                    // 如果下一个值与当前值相同，则向后移动直到找到不同的值
                    size_t next_pos = current_pos;
                    while (next_pos < values.size() - 1 && 
                           std::abs(values[next_pos] - values[current_pos]) < 1e-10) {
                        next_pos++;
                    }
                    
                    // 取当前值与下一不同值的中点作为切分点
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
    
    // 将单个特征值映射到对应的箱编号
    int getBin(double value, int feature_idx) const {
        if (feature_idx < 0 || feature_idx >= static_cast<int>(feature_cuts.size())) {
            return 0; // 默认箱
        }
        
        const std::vector<double>& cuts = feature_cuts[feature_idx];
        if (cuts.empty()) {
            return 0;
        }
        
        // 处理特殊情况
        if (std::isnan(value)) return 0;
        if (value <= cuts.front()) return 0;
        if (value > cuts.back()) return cuts.size();
        
        // 二分查找定位箱编号
        auto it = std::upper_bound(cuts.begin(), cuts.end(), value);
        return std::distance(cuts.begin(), it);
    }
    
    // 获取指定特征和箱的切分值（用于决策树分裂）
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
    std::vector<std::vector<double>> feature_cuts; // 每个特征的切分点
};

} // namespace Binning

#endif // BINNING_METHODS_H