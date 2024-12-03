#include "feature_importance.h"
#include "math_functions.h"
#include <algorithm>
#include <numeric>

std::map<int, double> FeatureImportance::calculateNodeImportances(
    const DecisionTreeSingle::Tree* node,
    double weighted_n_samples,
    double parent_mse) {
    
    std::map<int, double> importances;
    
    if (!node || node->IsLeaf) {
        return importances;
    }

    // Calcul de l'importance pour ce nœud
    double current_importance = (parent_mse - node->NodeMSE) * (node->NodeSamples / weighted_n_samples);
    importances[node->FeatureIndex] = current_importance;

    // Récursion sur les sous-arbres
    if (node->Left) {
        auto left_importances = calculateNodeImportances(node->Left.get(), weighted_n_samples, node->NodeMSE);
        for (const auto& [feature, importance] : left_importances) {
            importances[feature] += importance;
        }
    }
    
    if (node->Right) {
        auto right_importances = calculateNodeImportances(node->Right.get(), weighted_n_samples, node->NodeMSE);
        for (const auto& [feature, importance] : right_importances) {
            importances[feature] += importance;
        }
    }

    return importances;
}

std::vector<FeatureImportance::FeatureScore> FeatureImportance::calculateTreeImportance(
    const DecisionTreeSingle& tree,
    const std::vector<std::string>& feature_names) {
    
    std::vector<FeatureScore> feature_scores;
    auto importances = calculateNodeImportances(tree.getRoot(), tree.getRootSamples(), tree.getRootMSE());
    
    // Normalisation et création des scores
    double total_importance = 0.0;
    for (const auto& [_, importance] : importances) {
        total_importance += importance;
    }

    for (size_t i = 0; i < feature_names.size(); ++i) {
        double importance = importances.count(i) ? importances[i] / total_importance : 0.0;
        feature_scores.emplace_back(i, feature_names[i], importance);
    }

    // Tri par importance décroissante
    std::sort(feature_scores.begin(), feature_scores.end(),
              [](const FeatureScore& a, const FeatureScore& b) {
                  return a.importance_score > b.importance_score;
              });

    return feature_scores;
}

std::vector<FeatureImportance::FeatureScore> FeatureImportance::calculateBaggingImportance(
    const Bagging& model,
    const std::vector<std::string>& feature_names) {
    
    std::vector<double> total_importances(feature_names.size(), 0.0);
    
    // Calcul de la moyenne des importances sur tous les arbres
    for (const auto& tree : model.getTrees()) {
        auto tree_importances = calculateTreeImportance(*tree, feature_names);
        for (const auto& score : tree_importances) {
            total_importances[score.feature_index] += score.importance_score;
        }
    }

    // Normalisation
    double sum = std::accumulate(total_importances.begin(), total_importances.end(), 0.0);
    std::vector<FeatureScore> feature_scores;
    
    for (size_t i = 0; i < feature_names.size(); ++i) {
        double normalized_importance = sum > 0 ? total_importances[i] / sum : 0.0;
        feature_scores.emplace_back(i, feature_names[i], normalized_importance);
    }

    // Tri par importance décroissante
    std::sort(feature_scores.begin(), feature_scores.end(),
              [](const FeatureScore& a, const FeatureScore& b) {
                  return a.importance_score > b.importance_score;
              });

    return feature_scores;
}

std::vector<FeatureImportance::FeatureScore> FeatureImportance::calculateBoostingImportance(
    const Boosting& model,
    const std::vector<std::string>& feature_names) {
    
    std::vector<double> total_importances(feature_names.size(), 0.0);
    
    // Calcul de la moyenne pondérée des importances sur tous les arbres
    for (size_t i = 0; i < model.getEstimators().size(); ++i) {
        auto tree_importances = calculateTreeImportance(*model.getEstimators()[i], feature_names);
        double weight = 1.0 / (i + 1); // Poids décroissant pour les arbres plus récents
        
        for (const auto& score : tree_importances) {
            total_importances[score.feature_index] += score.importance_score * weight;
        }
    }

    // Normalisation
    double sum = std::accumulate(total_importances.begin(), total_importances.end(), 0.0);
    std::vector<FeatureScore> feature_scores;
    
    for (size_t i = 0; i < feature_names.size(); ++i) {
        double normalized_importance = sum > 0 ? total_importances[i] / sum : 0.0;
        feature_scores.emplace_back(i, feature_names[i], normalized_importance);
    }

    // Tri par importance décroissante
    std::sort(feature_scores.begin(), feature_scores.end(),
              [](const FeatureScore& a, const FeatureScore& b) {
                  return a.importance_score > b.importance_score;
              });

    return feature_scores;
} 