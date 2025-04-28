#ifndef BOOSTING_IMPROVED_H
#define BOOSTING_IMPROVED_H

#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <memory>  // Pour std::unique_ptr
#include "binning_methods.h"

// Classe GBDT améliorée avec méthodes de binning
class ImprovedGBDT {
public:
    // Méthodes de binning disponibles
    enum BinningMethod {
        NONE,           // Pas de binning
        QUANTILE,       // Quantile Sketch (méthode XGBoost)
        FREQUENCY       // Frequency Binning (méthode LightGBM)
    };

    struct Node {
        bool is_leaf;
        int split_feature;
        double split_value;
        double leaf_value;
        Node* left;
        Node* right;
        Node(): is_leaf(false), split_feature(-1), split_value(0.0), leaf_value(0.0), left(nullptr), right(nullptr) {}
    };

    struct Tree {
        Node* root;
        Tree(): root(nullptr) {}
    };

    // Constructeur avec paramètres
    ImprovedGBDT(int n_estimators, int max_depth, double learning_rate = 0.1,
                 bool useDart = false, double drop_rate = 0.1, double skip_rate = 0.0,
                 BinningMethod binning_method = NONE, int num_bins = 256)
        : n_estimators(n_estimators), max_depth(max_depth), learning_rate(learning_rate),
          useDart(useDart), drop_rate(drop_rate), skip_rate(skip_rate),
          binning_method(binning_method), num_bins(num_bins) {
        rng.seed(123);  // Seed fixe pour reproductibilité
        initial_prediction = 0.0;
    }

    // Destructeur pour libérer la mémoire
    ~ImprovedGBDT() {
        for (auto &tree : trees) {
            freeTree(tree.root);
        }
    }

    // Méthode d'entrainement
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    // Prédiction pour un seul échantillon
    double predict(const std::vector<double>& x) const;

    // Prédiction pour plusieurs échantillons
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

    // Objets de binning
    std::unique_ptr<Binning::QuantileSketch> quantile_binner;
    std::unique_ptr<Binning::FrequencyBinning> frequency_binner;

    // Construction récursive d'arbre (méthode standard)
    Node* buildTreeRecursive(const std::vector<std::vector<double>>& X, const std::vector<double>& residual, 
                             const std::vector<int>& indices, int depth);
    
        // Dans boosting_improved.h
    Node* buildTreeRecursiveBinned(const std::vector<std::vector<double>>& X, 
        const std::vector<double>& residual, 
        const std::vector<int>& indices, int depth,
        const std::vector<std::vector<double>>& parent_grad_sum = {},
        const std::vector<std::vector<int>>& parent_count = {});
    // Obtenir l'indice de bin pour une valeur donnée
    int getBinIndex(double value, int feature_idx) const;
    
    // Obtenir la valeur de séparation à partir d'un indice de bin
    double getSplitValueFromBin(int feature_idx, int bin_idx) const;

    // Libérer la mémoire de l'arbre
    void freeTree(Node* node) {
        if (!node) return;
        freeTree(node->left);
        freeTree(node->right);
        delete node;
    }
};

#endif // BOOSTING_IMPROVED_H