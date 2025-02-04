#ifndef FEATURE_IMPORTANCE_H
#define FEATURE_IMPORTANCE_H

#include "../tree/decision_tree_single.h"
#include "../../ensemble/bagging/bagging.h"
#include "../../ensemble/boosting/boosting.h"
#include "../math/math_functions.h"
#include <vector>
#include <string>
#include <map>

class FeatureImportance {
public:
    struct FeatureScore {
        int feature_index;
        std::string feature_name;
        double importance_score;
        
        // Constructor to make creation easier
        FeatureScore(int idx, const std::string& name, double score) 
            : feature_index(idx), feature_name(name), importance_score(score) {}
    };

    // Calculation of importance for a single tree
    static std::vector<FeatureScore> calculateTreeImportance(
        const DecisionTreeSingle& tree,
        const std::vector<std::string>& feature_names);

    // Calculation of importance for bagging
    static std::vector<FeatureScore> calculateBaggingImportance(
        const Bagging& model,
        const std::vector<std::string>& feature_names);

    // Calculation of importance for boosting
    static std::vector<FeatureScore> calculateBoostingImportance(
        const Boosting& model,
        const std::vector<std::string>& feature_names);

private:
    static std::map<int, double> calculateNodeImportances(
        const DecisionTreeSingle::Tree* node,
        double weighted_n_samples,
        double parent_mse);
};

#endif // FEATURE_IMPORTANCE_H
