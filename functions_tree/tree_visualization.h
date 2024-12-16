#ifndef TREE_VISUALIZATION_H
#define TREE_VISUALIZATION_H

#include "decision_tree_single.h"
#include <string>
#include <vector>
#include <fstream>

class TreeVisualization {
public:
    // Générer le fichier DOT pour un arbre unique
    static void generateDotFile(const DecisionTreeSingle& tree,
                              const std::string& filename,
                              const std::vector<std::string>& feature_names);

    // Générer le fichier DOT pour un ensemble d'arbres (bagging ou boosting)
    static void generateEnsembleDotFiles(const std::vector<std::unique_ptr<DecisionTreeSingle>>& trees,
                                       const std::string& base_filename,
                                       const std::vector<std::string>& feature_names);

private:
    // Générer le contenu DOT pour un nœud
    static void generateDotContent(const DecisionTreeSingle::Tree* node,
                                 std::ofstream& out,
                                 int& node_count,
                                 const std::vector<std::string>& feature_names,
                                 int depth,
                                 int& total_nodes);

    // Formater un nœud pour l'affichage
    static std::string formatNode(const DecisionTreeSingle::Tree* node,
                                const std::vector<std::string>& feature_names);
};

#endif // TREE_VISUALIZATION_H 