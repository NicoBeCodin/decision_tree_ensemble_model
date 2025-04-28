#include "boosting_improved.h"
#include <omp.h>
#include <memory>

// Implémentation de la méthode d'entrainement
void ImprovedGBDT::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    int n_samples = X.size();
    if (n_samples == 0) return;
    int n_features = X[0].size();

    // Prétraitement: construction des bins selon la méthode choisie
    if (binning_method == QUANTILE) {
        quantile_binner = std::make_unique<Binning::QuantileSketch>(num_bins);
        quantile_binner->build(X);
    } else if (binning_method == FREQUENCY) {
        frequency_binner = std::make_unique<Binning::FrequencyBinning>(num_bins);
        frequency_binner->build(X);
    }

    // Initialisation de la prédiction avec la moyenne des valeurs cibles
    double sumY = 0.0;
    for (double val : y) {
        sumY += val;
    }
    initial_prediction = sumY / n_samples;
    y_pred_train.assign(n_samples, initial_prediction);

    // Allocation préalable pour optimisation
    trees.reserve(n_estimators);
    tree_weights.reserve(n_estimators);

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Vecteurs pour les gradients et hessians
    std::vector<double> grad(n_samples);
    std::vector<double> hess(n_samples);

    // Boucle principale d'entrainement
    for (int iter = 0; iter < n_estimators; ++iter) {
        // Gestion du skip rate pour DART
        if (useDart && skip_rate > 0.0) {
            double skip_sample = dist(rng);
            if (skip_sample < skip_rate) {
                continue; // On saute cette itération
            }
        }

        // Calcul des prédictions pour le gradient (avec gestion DART)
        std::vector<int> drop_indices;
        std::vector<double> pred_for_grad;
        
        if (useDart) {
            // Mode DART: sélection aléatoire des arbres à ignorer
            if (!trees.empty() && drop_rate > 0.0) {
                for (int j = 0; j < (int)trees.size(); ++j) {
                    double r = dist(rng);
                    if (r < drop_rate) {
                        drop_indices.push_back(j);
                    }
                }
                // On garde au moins un arbre
                if (drop_indices.size() == trees.size()) {
                    drop_indices.pop_back();
                }
            }
            
            // Si des arbres sont ignorés, recalcul des prédictions sans eux
            if (!drop_indices.empty()) {
                pred_for_grad = y_pred_train;
                for (int drop_idx : drop_indices) {
                    Node* node = trees[drop_idx].root;
                    for (int i = 0; i < n_samples; ++i) {
                        double tree_pred = 0.0;
                        Node* cur = node;
                        while (cur && !cur->is_leaf) {
                            if (X[i][cur->split_feature] <= cur->split_value) {
                                cur = cur->left;
                            } else {
                                cur = cur->right;
                            }
                        }
                        if (cur) {
                            tree_pred = cur->leaf_value;
                        }
                        pred_for_grad[i] -= tree_weights[drop_idx] * tree_pred;
                    }
                }
            } else {
                pred_for_grad = y_pred_train;
            }
        } else {
            pred_for_grad = y_pred_train;
        }

        // Calcul des gradients et hessians (pour MSE)
        for (int i = 0; i < n_samples; ++i) {
            double error = pred_for_grad[i] - y[i];
            grad[i] = error;
            hess[i] = 1.0;
        }

        // Calcul des résidus (négatif du gradient)
        std::vector<double> residual(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            residual[i] = -grad[i];
        }
        
        // Indices de tous les échantillons
        std::vector<int> all_indices(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            all_indices[i] = i;
        }
        
        // Construction de l'arbre selon la méthode choisie
        Node* root = nullptr;
        if (binning_method != NONE) {
            root = buildTreeRecursiveBinned(X, residual, all_indices, 0);
        } else {
            root = buildTreeRecursive(X, residual, all_indices, 0);
        }
        
        Tree new_tree;
        new_tree.root = root;

        // Calcul des prédictions du nouvel arbre
        std::vector<double> new_tree_pred(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            Node* cur = root;
            while (cur && !cur->is_leaf) {
                if (X[i][cur->split_feature] <= cur->split_value) {
                    cur = cur->left;
                } else {
                    cur = cur->right;
                }
            }
            if (cur) {
                new_tree_pred[i] = cur->leaf_value;
            } else {
                new_tree_pred[i] = 0.0;
            }
        }

        // Ajout du nouvel arbre au modèle
        trees.push_back(new_tree);
        double new_weight = learning_rate;
        tree_weights.push_back(new_weight);

        // Mise à jour des prédictions
        if (useDart) {
            if (!drop_indices.empty()) {
                // Normalisation des poids (DART)
                double sumWeight = 0.0;
                for (double w : tree_weights) {
                    sumWeight += w;
                }
                double old_sum = sumWeight - tree_weights.back();
                double factor = (old_sum > 0.0 ? old_sum / sumWeight : 1.0);
                
                // Mise à l'échelle des poids
                for (double &w : tree_weights) {
                    w *= factor;
                }
                
                // Mise à jour des prédictions
                for (int i = 0; i < n_samples; ++i) {
                    y_pred_train[i] = factor * y_pred_train[i] + tree_weights.back() * new_tree_pred[i];
                }
            } else {
                // Sans drop, mise à jour standard
                for (int i = 0; i < n_samples; ++i) {
                    y_pred_train[i] += new_weight * new_tree_pred[i];
                }
            }
        } else {
            // Mode non-DART: mise à jour standard
            for (int i = 0; i < n_samples; ++i) {
                y_pred_train[i] += new_weight * new_tree_pred[i];
            }
        }
    }
}

// Obtenir l'indice de bin pour une valeur
int ImprovedGBDT::getBinIndex(double value, int feature_idx) const {
    if (binning_method == QUANTILE && quantile_binner) {
        return quantile_binner->getBin(value, feature_idx);
    } else if (binning_method == FREQUENCY && frequency_binner) {
        return frequency_binner->getBin(value, feature_idx);
    }
    return 0; // Valeur par défaut
}

// Obtenir la valeur de séparation à partir d'un indice de bin
double ImprovedGBDT::getSplitValueFromBin(int feature_idx, int bin_idx) const {
    if (binning_method == QUANTILE && quantile_binner) {
        return quantile_binner->getSplitValue(feature_idx, bin_idx);
    } else if (binning_method == FREQUENCY && frequency_binner) {
        return frequency_binner->getSplitValue(feature_idx, bin_idx);
    }
    return 0.0; // Valeur par défaut
}

ImprovedGBDT::Node* ImprovedGBDT::buildTreeRecursiveBinned(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& residual, 
    const std::vector<int>& indices, 
    int depth,
    const std::vector<std::vector<double>>& parent_grad_sum,
    const std::vector<std::vector<int>>& parent_count){
    
    if (indices.empty()) {
        return nullptr;
    }
    
    // Critères d'arrêt
    if (depth >= max_depth || indices.size() <= 1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        double sum = 0.0;
        for (int idx : indices) {
            sum += residual[idx];
        }
        leaf->leaf_value = sum / indices.size();
        return leaf;
    }
    
    // Calcul de la variance des résidus
    double sumRes = 0.0;
    double sumResSq = 0.0;
    for (int idx : indices) {
        sumRes += residual[idx];
        sumResSq += residual[idx] * residual[idx];
    }
    double meanRes = sumRes / indices.size();
    double varRes = sumResSq / indices.size() - meanRes * meanRes;
    
    if (varRes < 1e-9) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = meanRes;
        return leaf;
    }

    // Recherche de la meilleure division avec les bins
    double best_gain = 0.0;
    int best_feature = -1;
    int best_bin = 0;
    double best_split_value = 0.0;
    
    // Structure pour stocker le meilleur split
    std::vector<int> best_left_indices;
    std::vector<int> best_right_indices;
    
    // Stockage des histogrammes pour le nœud courant
    std::vector<std::vector<double>> node_grad_sum(X[0].size(), std::vector<double>(num_bins + 1, 0.0));
    std::vector<std::vector<int>> node_count(X[0].size(), std::vector<int>(num_bins + 1, 0));

    // Calcul des statistiques du nœud actuel
    double total_grad = 0.0;
    double total_hess = static_cast<double>(indices.size());
    
    // Utiliser histogramme parent si disponible, sinon construire un nouveau
    bool use_parent_hist = !parent_grad_sum.empty() && !parent_count.empty();
    
    if (!use_parent_hist) {
        // Construction complète de l'histogramme
        for (int idx : indices) {
            total_grad += -residual[idx];
            
            // Classifier dans les bins
            for (int f = 0; f < (int)X[0].size(); ++f) {
                int bin = getBinIndex(X[idx][f], f);
                node_grad_sum[f][bin] += -residual[idx];
                node_count[f][bin] += 1;
            }
        }
    } else {
        // Récupérer statistiques globales du parent
        total_grad = 0.0;
        for (int f = 0; f < (int)X[0].size(); ++f) {
            for (int bin = 0; bin <= num_bins; ++bin) {
                node_grad_sum[f][bin] = parent_grad_sum[f][bin];
                node_count[f][bin] = parent_count[f][bin];
                total_grad += node_grad_sum[f][bin];
            }
        }
    }
    
    int feature_count = X[0].size();
    
    // Pour chaque caractéristique
    for (int f = 0; f < feature_count; ++f) {
        // Chercher la meilleure division
        double left_grad_sum = 0.0;
        int left_count = 0;
        
        for (int bin = 0; bin < num_bins; ++bin) {
            if (node_count[f][bin] == 0) continue;  // Bin vide
            
            left_grad_sum += node_grad_sum[f][bin];
            left_count += node_count[f][bin];
            
            double right_grad_sum = total_grad - left_grad_sum;
            int right_count = indices.size() - left_count;
            
            if (left_count < 1 || right_count < 1) continue;
            
            // Calcul du gain de la division
            double gain = (left_grad_sum * left_grad_sum) / left_count
                        + (right_grad_sum * right_grad_sum) / right_count
                        - (total_grad * total_grad) / total_hess;
            
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_bin = bin;
                best_split_value = getSplitValueFromBin(f, bin);
            }
        }
    }
    
    // Si aucune division améliorante n'est trouvée
    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = meanRes;
        return leaf;
    }
    
    // Division des données selon la meilleure division
    best_left_indices.clear();
    best_right_indices.clear();
    
    for (int idx : indices) {
        if (X[idx][best_feature] <= best_split_value) {
            best_left_indices.push_back(idx);
        } else {
            best_right_indices.push_back(idx);
        }
    }
    
    // Histogrammes pour les enfants (seulement si assez grand pour justifier l'optimisation)
    std::vector<std::vector<double>> left_grad_sum(feature_count);
    std::vector<std::vector<double>> right_grad_sum(feature_count);
    std::vector<std::vector<int>> left_count(feature_count);
    std::vector<std::vector<int>> right_count(feature_count);
    
    const int MIN_NODE_SIZE_FOR_SUBTRACTION = 50; // Seuil arbitraire
    
    // Préparer les histogrammes des enfants par soustraction si le nœud est assez grand
    if (indices.size() > MIN_NODE_SIZE_FOR_SUBTRACTION) {
        // Construire l'histogramme de l'enfant gauche
        std::vector<std::vector<double>> left_hist_grad(feature_count, std::vector<double>(num_bins + 1, 0.0));
        std::vector<std::vector<int>> left_hist_count(feature_count, std::vector<int>(num_bins + 1, 0));
        
        // On ne scanne que les échantillons de gauche (plus petit ensemble)
        for (int idx : best_left_indices) {
            for (int f = 0; f < feature_count; ++f) {
                int bin = getBinIndex(X[idx][f], f);
                left_hist_grad[f][bin] += -residual[idx];
                left_hist_count[f][bin] += 1;
            }
        }
        
        // Par soustraction, calculer l'histogramme de droite
        std::vector<std::vector<double>> right_hist_grad(feature_count, std::vector<double>(num_bins + 1, 0.0));
        std::vector<std::vector<int>> right_hist_count(feature_count, std::vector<int>(num_bins + 1, 0));
        
        for (int f = 0; f < feature_count; ++f) {
            for (int bin = 0; bin <= num_bins; ++bin) {
                right_hist_grad[f][bin] = node_grad_sum[f][bin] - left_hist_grad[f][bin];
                right_hist_count[f][bin] = node_count[f][bin] - left_hist_count[f][bin];
            }
        }
        
        left_grad_sum = std::move(left_hist_grad);
        right_grad_sum = std::move(right_hist_grad);
        left_count = std::move(left_hist_count);
        right_count = std::move(right_hist_count);
    }
    
    // Création du nœud et construction récursive des sous-arbres
    Node* node = new Node();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_value = best_split_value;
    
    // Passer les histogrammes aux enfants si disponibles
    if (indices.size() > MIN_NODE_SIZE_FOR_SUBTRACTION) {
        node->left = buildTreeRecursiveBinned(X, residual, best_left_indices, depth + 1, 
                                            left_grad_sum, left_count);
        node->right = buildTreeRecursiveBinned(X, residual, best_right_indices, depth + 1, 
                                             right_grad_sum, right_count);
    } else {
        // Sinon, construire normalement (pour les petits nœuds)
        node->left = buildTreeRecursiveBinned(X, residual, best_left_indices, depth + 1);
        node->right = buildTreeRecursiveBinned(X, residual, best_right_indices, depth + 1);
    }
    
    // Vérification et correction
    if (node->left == nullptr && node->right == nullptr) {
        node->is_leaf = true;
        node->leaf_value = meanRes;
        node->left = node->right = nullptr;
    }
    
    return node;
}

// Construction récursive d'arbre standard
ImprovedGBDT::Node* ImprovedGBDT::buildTreeRecursive(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& residual, 
    const std::vector<int>& indices, int depth) {
    
    if (indices.empty()) {
        return nullptr;
    }
    
    // Critères d'arrêt
    if (depth >= max_depth || indices.size() <= 1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        double sum = 0.0;
        for (int idx : indices) {
            sum += residual[idx];
        }
        leaf->leaf_value = sum / indices.size();
        return leaf;
    }
    
    // Calcul de la variance des résidus
    double sumRes = 0.0;
    double sumResSq = 0.0;
    for (int idx : indices) {
        sumRes += residual[idx];
        sumResSq += residual[idx] * residual[idx];
    }
    double meanRes = sumRes / indices.size();
    double varRes = sumResSq / indices.size() - meanRes * meanRes;
    
    if (varRes < 1e-9) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = meanRes;
        return leaf;
    }

    // Recherche de la meilleure division
    double best_gain = 0.0;
    int best_feature = -1;
    double best_split_value = 0.0;
    std::vector<int> best_left_indices;
    std::vector<int> best_right_indices;

    // Calcul des statistiques du noeud actuel
    double total_grad = 0.0;
    double total_hess = static_cast<double>(indices.size());
    for (int idx : indices) {
        total_grad += -residual[idx];
    }

    int feature_count = X[0].size();
    
    // Pour chaque caractéristique
    for (int f = 0; f < feature_count; ++f) {
        // Tri des indices selon la caractéristique
        std::vector<int> sorted_idx = indices;
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b){
            return X[a][f] < X[b][f];
        });
        
        double left_grad_sum = 0.0;
        double left_count = 0.0;
        
        // Essai de chaque point de division possible
        for (size_t i = 0; i < sorted_idx.size() - 1; ++i) {
            int idx = sorted_idx[i];
            double gradVal = -residual[idx];
            left_grad_sum += gradVal;
            left_count += 1.0;
            
            // Ne pas diviser entre valeurs identiques
            if (X[sorted_idx[i]][f] == X[sorted_idx[i+1]][f]) {
                continue;
            }
            
            // Calcul des statistiques de division
            double right_grad_sum = total_grad - left_grad_sum;
            double right_count = total_hess - left_count;
            
            if (left_count < 1e-6 || right_count < 1e-6) {
                continue;
            }
            
            // Calcul du gain de division
            double gain = (left_grad_sum * left_grad_sum) / left_count 
                        + (right_grad_sum * right_grad_sum) / right_count 
                        - (total_grad * total_grad) / total_hess;
            
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                
                // Point de division = moyenne entre la valeur actuelle et la suivante
                double val = X[sorted_idx[i]][f];
                double next_val = X[sorted_idx[i+1]][f];
                best_split_value = (val + next_val) / 2.0;
                
                best_left_indices.assign(sorted_idx.begin(), sorted_idx.begin() + i + 1);
                best_right_indices.assign(sorted_idx.begin() + i + 1, sorted_idx.end());
            }
        }
    }

    // Si aucune division améliorante n'est trouvée
    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->leaf_value = meanRes;
        return leaf;
    }

    // Création du noeud et construction récursive des sous-arbres
    Node* node = new Node();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_value = best_split_value;
    node->left = buildTreeRecursive(X, residual, best_left_indices, depth + 1);
    node->right = buildTreeRecursive(X, residual, best_right_indices, depth + 1);
    
    // Vérification et correction
    if (node->left == nullptr && node->right == nullptr) {
        node->is_leaf = true;
        node->leaf_value = meanRes;
        node->left = node->right = nullptr;
    }
    
    return node;
}

// Prédiction pour un seul échantillon
double ImprovedGBDT::predict(const std::vector<double>& x) const {
    double pred = initial_prediction;
    
    // Addition des prédictions pondérées de chaque arbre
    for (size_t j = 0; j < trees.size(); ++j) {
        Node* node = trees[j].root;
        while (node && !node->is_leaf) {
            if (x[node->split_feature] <= node->split_value) {
                node = node->left;
            } else {
                node = node->right;
            }
        }
        double leaf_val = node ? node->leaf_value : 0.0;
        pred += tree_weights[j] * leaf_val;
    }
    
    return pred;
}

// Prédiction pour plusieurs échantillons
std::vector<double> ImprovedGBDT::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> preds;
    preds.reserve(X.size());
    
    for (const auto& x : X) {
        preds.push_back(predict(x));
    }
    
    return preds;
}