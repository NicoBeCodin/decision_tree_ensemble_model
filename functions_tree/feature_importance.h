#ifndef FEATURE_IMPORTANCE_H
#define FEATURE_IMPORTANCE_H

#include "decision_tree_single.h"
#include "../ensemble_bagging/bagging.h"
#include "../ensemble_boosting/boosting.h"
#include <vector>
#include <string>
#include <map>

class FeatureImportance {
public:
    struct FeatureScore {
        int feature_index;
        std::string feature_name;
        double importance_score;
        
        // Constructeur pour faciliter la création
        FeatureScore(int idx, const std::string& name, double score) 
            : feature_index(idx), feature_name(name), importance_score(score) {}
    };

    // Calcul de l'importance pour un arbre unique
    static std::vector<FeatureScore> calculateTreeImportance(
        const DecisionTreeSingle& tree,
        const std::vector<std::string>& feature_names);

    // Calcul de l'importance pour le bagging
    static std::vector<FeatureScore> calculateBaggingImportance(
        const Bagging& model,
        const std::vector<std::string>& feature_names);

    // Calcul de l'importance pour le boosting
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