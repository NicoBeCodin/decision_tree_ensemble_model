#ifndef TREE_VISUALIZATION_H
#define TREE_VISUALIZATION_H

#include "decision_tree_single.h"
#include <string>
#include <vector>
#include <fstream>

class TreeVisualization {
public:
    // Generate the DOT file for a single tree
    static void generateDotFile(const DecisionTreeSingle& tree,
                              const std::string& filename,
                              const std::vector<std::string>& feature_names,
                              int criteria);

    // Generate the DOT file for an ensemble of trees (bagging or boosting)
    static void generateEnsembleDotFiles(const std::vector<std::unique_ptr<DecisionTreeSingle>>& trees,
                                       const std::string& base_filename,
                                       const std::vector<std::string>& feature_names,
                                       int criteria);

private:
    // Generate the DOT content for a node
    static void generateDotContent(const DecisionTreeSingle::Tree* node,
                                 std::ofstream& out,
                                 int& node_count,
                                 const std::vector<std::string>& feature_names,
                                 int depth,
                                 int& total_nodes,
                                 int criteria);

    // Format a node for display
    static std::string formatNode(const DecisionTreeSingle::Tree* node,
                                const std::vector<std::string>& feature_names,  int criteria);
};

#endif // TREE_VISUALIZATION_H
