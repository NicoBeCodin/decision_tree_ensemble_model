    #include "bagging.h"
    #include <random>
    #include <algorithm>
    #include <stdexcept>
    #include <fstream>

    /**
    * Constructor for Bagging
    * @param num_trees Number of trees in the Bagging ensemble
    * @param max_depth Maximum depth of each tree
    * @param min_samples_split Minimum number of samples required to split a node
    * @param min_impurity_decrease Minimum impurity decrease required for a split
    */
    Bagging::Bagging(int num_trees, int max_depth, int min_samples_split, double min_impurity_decrease)
        : numTrees(num_trees), maxDepth(max_depth), minSamplesSplit(min_samples_split), minImpurityDecrease(min_impurity_decrease) {
        trees.reserve(numTrees); // Reserve space for the trees
    }

    /**
    * Generate a bootstrap sample from the dataset
    * @param data Original dataset's feature matrix
    * @param labels Original dataset's target vector
    * @param sampled_data Output parameter for the sampled feature matrix
    * @param sampled_labels Output parameter for the sampled target vector
    */
    void Bagging::bootstrapSample(const std::vector<std::vector<double>>& data, const std::vector<double>& labels,
                                std::vector<std::vector<double>>& sampled_data, std::vector<double>& sampled_labels) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.size() - 1);

        size_t n_samples = data.size();
        sampled_data.reserve(n_samples);
        sampled_labels.reserve(n_samples);

        for (size_t i = 0; i < n_samples; ++i) {
            int idx = dis(gen);
            sampled_data.push_back(data[idx]);
            sampled_labels.push_back(labels[idx]);
        }
    }

    /**
    * Train the Bagging ensemble
    * @param data Feature matrix
    * @param labels Target vector
    */
    void Bagging::train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels) {
        for (int i = 0; i < numTrees; ++i) {
            std::vector<std::vector<double>> sampled_data;
            std::vector<double> sampled_labels;
            bootstrapSample(data, labels, sampled_data, sampled_labels);

            // Create and train a new DecisionTreeSingle
            auto tree = std::make_unique<DecisionTreeSingle>(maxDepth, minSamplesSplit, minImpurityDecrease);
            tree->train(sampled_data, sampled_labels, 0);
            trees.push_back(std::move(tree));
        }
    }

    /**
    * Predict the target value for a single sample
    * @param sample Feature vector for the sample
    * @return Averaged prediction from all trees
    */
    double Bagging::predict(const std::vector<double>& sample) const {
        double sum = 0.0;
        for (const auto& tree : trees) {
            sum += tree->predict(sample);
        }
        return sum / trees.size(); // Return the average prediction
    }

    /**
    * Evaluate the Bagging model on a test dataset
    * @param test_data Feature matrix of the test set
    * @param test_labels Target vector of the test set
    * @return Mean Squared Error (MSE) on the test set
    */
    double Bagging::evaluate(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_labels) const {
        double total_error = 0.0;
        for (size_t i = 0; i < test_data.size(); ++i) {
            double prediction = predict(test_data[i]);
            double diff = prediction - test_labels[i];
            total_error += diff * diff;
        }
        return total_error / test_data.size(); // Return the Mean Squared Error
    }

    void Bagging::save(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        // Sauvegarder les paramètres du modèle
        file << numTrees << " " << maxDepth << " " << minSamplesSplit << " " << minImpurityDecrease << "\n";
        
        // Sauvegarder chaque arbre avec un nom unique
        for (size_t i = 0; i < trees.size(); ++i) {
            std::string tree_filename = filename + "_tree_" + std::to_string(i);
            trees[i]->saveTree(tree_filename);
        }
        
        file.close();
    }

    void Bagging::load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
        
        // Vérifier la validité du fichier
        int num_trees, max_depth, min_samples_split;
        double min_impurity_decrease;
        if (!(file >> num_trees >> max_depth >> min_samples_split >> min_impurity_decrease)) {
            file.close();
            throw std::runtime_error("Invalid file format");
        }
        
        // Mettre à jour les paramètres du modèle
        numTrees = num_trees;
        maxDepth = max_depth;
        minSamplesSplit = min_samples_split;
        minImpurityDecrease = min_impurity_decrease;
        
        // Réinitialiser et recharger les arbres
        trees.clear();
        trees.resize(numTrees);
        
        // Charger chaque arbre
        for (int i = 0; i < numTrees; ++i) {
            std::string tree_filename = filename + "_tree_" + std::to_string(i);
            trees[i] = std::make_unique<DecisionTreeSingle>(maxDepth, minSamplesSplit, minImpurityDecrease);
            trees[i]->loadTree(tree_filename);
        }
        
        file.close();
    }
