#include "tree_visualization.h"
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <chrono>

// Constants to limit the size of the visualizations
const int MAX_DEPTH_VISUALIZATION = 5;  // Maximum depth for visualization
const int MAX_NODES_VISUALIZATION = 50;  // Maximum number of nodes to display

void TreeVisualization::generateDotFile(const DecisionTreeSingle& tree,
                                        const std::string& filename,
                                        const std::vector<std::string>& feature_names,
                                        int criteria) {
    try {
        if (!tree.getRoot()) {
            std::cout << "The tree is empty, unable to generate visualization." << std::endl;
            return;
        }

        // Create the folder for DOT files
        std::filesystem::path dotPath = "visualizations/dot";
        if (!std::filesystem::exists(dotPath)) {
            std::filesystem::create_directories(dotPath);
        }

        // Create the folder for PNG images
        std::filesystem::path pngPath = std::filesystem::current_path().parent_path() / "tree_visualizations";
        if (!std::filesystem::exists(pngPath)) {
            std::filesystem::create_directories(pngPath);
        }

        // Full paths for the files
        auto dot_file = std::filesystem::absolute(dotPath / (filename + ".dot"));
        auto png_file = std::filesystem::absolute(pngPath / (filename + ".png"));

        // Create the DOT file
        std::ofstream out(dot_file);
        if (!out.is_open()) {
            std::cout << "Unable to open the DOT file for writing" << std::endl;
            return;
        }

        // DOT file header
        out << "digraph DecisionTree {\n";
        out << "    rankdir=TB;\n";
        out << "    node [shape=box, style=\"rounded,filled\", color=black, fontname=helvetica, fontsize=10];\n";
        out << "    edge [fontname=helvetica, fontsize=9];\n";
        out << "    graph [ranksep=0.3, nodesep=0.3];\n";

        // Generate the tree content
        int node_count = 0;
        int total_nodes = 0;
        generateDotContent(tree.getRoot(), out, node_count, feature_names, 0, total_nodes, criteria);

        out << "}\n";
        out.close();

        // Generate the PNG image with Graphviz
        std::stringstream cmd;
        std::string dot_path = getenv("DOT_PATH") ? getenv("DOT_PATH") : "dot";
        cmd << dot_path << " -Tpng -Gdpi=150 \"" << dot_file.string() << "\" -o \"" << png_file.string() << "\"";

        int result = system(cmd.str().c_str());
        if (result != 0) {
            std::cout << "Error generating the PNG (code: " << result << "). Make sure Graphviz is installed and the path is correct." << std::endl;
        } else {
            std::cout << "Image generated: " << png_file.string() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

void TreeVisualization::generateEnsembleDotFiles(
    const std::vector<std::unique_ptr<DecisionTreeSingle>>& trees,
    const std::string& base_filename,
    const std::vector<std::string>& feature_names,
    int criteria) {
    
    std::cout << "Generating representative trees..." << std::endl;
    
    // Indices of the trees to visualize
    std::vector<size_t> important_indices;
    if (trees.size() > 0) {
        important_indices.push_back(0);  // First tree
    }
    if (trees.size() > 2) {
        important_indices.push_back(trees.size() / 2);  // Middle tree
    }
    if (trees.size() > 1) {
        important_indices.push_back(trees.size() - 1);  // Last tree
    }

    for (size_t idx : important_indices) {
        std::string suffix;
        if (idx == 0) suffix = "_first";
        else if (idx == trees.size() - 1) suffix = "_last";
        else suffix = "_middle";

        std::string filename = base_filename + suffix;
        std::cout << "\nGenerating tree " << (idx + 1) << "/" << trees.size() 
                  << " (position: " << suffix.substr(1) << ")" << std::endl;
        generateDotFile(*trees[idx], filename, feature_names, criteria);
    }

    std::cout << "Visualizations generated in the 'visualizations' folder" << std::endl;
}

void TreeVisualization::generateDotFileXGBoost(const DecisionTreeXGBoost& tree,
                                               const std::string& filename,
                                               const std::vector<std::string>& feature_names) {
    try {
        if (!tree.getRoot()) {
            std::cout << "The tree is empty, unable to generate visualization." << std::endl;
            return;
        }

        // Create folders for DOT and PNG files
        std::filesystem::path dotPath = "visualizations/dot";
        if (!std::filesystem::exists(dotPath)) {
            std::filesystem::create_directories(dotPath);
        }

        std::filesystem::path pngPath = std::filesystem::current_path().parent_path() / "tree_visualizations";
        if (!std::filesystem::exists(pngPath)) {
            std::filesystem::create_directories(pngPath);
        }

        // File paths
        auto dot_file = std::filesystem::absolute(dotPath / (filename + ".dot"));
        auto png_file = std::filesystem::absolute(pngPath / (filename + ".png"));

        // Create DOT file
        std::ofstream out(dot_file);
        if (!out.is_open()) {
            std::cout << "Unable to open the DOT file for writing" << std::endl;
            return;
        }

        // DOT header
        out << "digraph DecisionTree {\n";
        out << "    rankdir=TB;\n";
        out << "    node [shape=box, style=\"rounded,filled\", color=black, fontname=helvetica, fontsize=10];\n";
        out << "    edge [fontname=helvetica, fontsize=9];\n";
        out << "    graph [ranksep=0.3, nodesep=0.3];\n";

        // Generate content
        int node_count = 0;
        int total_nodes = 0;
        generateDotContentXGBoost(tree.getRoot(), out, node_count, feature_names, 0, total_nodes);

        out << "}\n";
        out.close();

        // Generate PNG using Graphviz
        std::stringstream cmd;
        std::string dot_path = getenv("DOT_PATH") ? getenv("DOT_PATH") : "dot";
        cmd << dot_path << " -Tpng -Gdpi=150 \"" << dot_file.string() << "\" -o \"" << png_file.string() << "\"";

        int result = system(cmd.str().c_str());
        if (result != 0) {
            std::cout << "Error generating the PNG (code: " << result << "). Make sure Graphviz is installed and the path is correct." << std::endl;
        } else {
            std::cout << "Image generated: " << png_file.string() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

void TreeVisualization::generateEnsembleDotFilesXGBoost(
    const std::vector<std::unique_ptr<DecisionTreeXGBoost>>& trees,
    const std::string& base_filename,
    const std::vector<std::string>& feature_names) {
    
    std::cout << "Generating representative trees..." << std::endl;
    
    // Indices of trees to visualize
    std::vector<size_t> important_indices;
    if (trees.size() > 0) {
        important_indices.push_back(0);  // First tree
    }
    if (trees.size() > 2) {
        important_indices.push_back(trees.size() / 2);  // Middle tree
    }
    if (trees.size() > 1) {
        important_indices.push_back(trees.size() - 1);  // Last tree
    }

    for (size_t idx : important_indices) {
        std::string suffix;
        if (idx == 0) suffix = "_first";
        else if (idx == trees.size() - 1) suffix = "_last";
        else suffix = "_middle";

        std::string filename = base_filename + suffix;
        std::cout << "\nGenerating tree " << (idx + 1) << "/" << trees.size() 
                  << " (position: " << suffix.substr(1) << ")" << std::endl;
        generateDotFileXGBoost(*trees[idx], filename, feature_names);
    }

    std::cout << "Visualizations generated in the 'visualizations' folder" << std::endl;
}

void TreeVisualization::generateDotContent(const DecisionTreeSingle::Tree* node,
                                         std::ofstream& out,
                                         int& node_count,
                                         const std::vector<std::string>& feature_names,
                                         int depth,
                                         int& total_nodes,
                                         int criteria) {
    if (!node || depth >= MAX_DEPTH_VISUALIZATION || total_nodes >= MAX_NODES_VISUALIZATION) return;

    int current_node = node_count++;
    total_nodes++;
    
    std::string node_label = formatNode(node, feature_names, criteria);
    std::string node_color = node->IsLeaf ? "lightblue" : "lightgreen";
    out << "    node" << current_node << " [label=\"" << node_label << "\", fillcolor=" << node_color << "];\n";

    if (!node->IsLeaf && depth < MAX_DEPTH_VISUALIZATION - 1) {
        int left_child = node_count;
        generateDotContent(node->Left.get(), out, node_count, feature_names, depth + 1, total_nodes, criteria);
        out << "    node" << current_node << " -> node" << left_child 
            << " [label=\"≤" << std::fixed << std::setprecision(1) << node->MaxValue << "\"];\n";

        int right_child = node_count;
        generateDotContent(node->Right.get(), out, node_count, feature_names, depth + 1, total_nodes, criteria);
        out << "    node" << current_node << " -> node" << right_child 
            << " [label=\">" << std::fixed << std::setprecision(1) << node->MaxValue << "\"];\n";
    }
}


std::string TreeVisualization::formatNode(const DecisionTreeSingle::Tree* node,
                                        const std::vector<std::string>& feature_names, int criteria) {
    std::stringstream ss;

    if (node->IsLeaf) {
        std::cout << "OEOEOEOOEOOEOEO" << std::endl;
        ss << std::fixed << std::setprecision(6);  // More precision for leaves
        ss << "Pred: " << node->Prediction;
    } else {
        std::string feature_name = (node->FeatureIndex < feature_names.size()) 
            ? feature_names[node->FeatureIndex] 
            : "f" + std::to_string(node->FeatureIndex);
        ss << feature_name << "\\n";
        ss << std::scientific << std::setprecision(4);  // Scientific notation for MSE
        if(criteria == 0) {
            ss << "MSE: " << node->NodeMetric<< "\\n";
        }
        if(criteria == 1) {
            ss << "MAE: " << node->NodeMetric<< "\\n";
        }
        ss << "N: " << node->NodeSamples;
    }

    return ss.str();
}

void TreeVisualization::generateDotContentXGBoost(const DecisionTreeXGBoost::Tree* node,
                                           std::ofstream& out,
                                           int& node_count,
                                           const std::vector<std::string>& feature_names,
                                           int depth,
                                           int& total_nodes) {
    if (!node || depth >= MAX_DEPTH_VISUALIZATION || total_nodes >= MAX_NODES_VISUALIZATION) return;

    int current_node = node_count++;
    total_nodes++;

    std::string node_label = formatNodeXGBoost(node, feature_names);
    std::string node_color = node->IsLeaf ? "lightblue" : "lightgreen";
    out << "    node" << current_node << " [label=\"" << node_label << "\", fillcolor=" << node_color << "];\n";

    if (!node->IsLeaf && depth < MAX_DEPTH_VISUALIZATION - 1) {
        int left_child = node_count;
        generateDotContentXGBoost(node->Left.get(), out, node_count, feature_names, depth + 1, total_nodes);
        out << "    node" << current_node << " -> node" << left_child << " [label=\"≤" 
            << std::fixed << std::setprecision(1) << node->MaxValue << "\"];\n";

        int right_child = node_count;
        generateDotContentXGBoost(node->Right.get(), out, node_count, feature_names, depth + 1, total_nodes);
        out << "    node" << current_node << " -> node" << right_child << " [label=\">" 
            << std::fixed << std::setprecision(1) << node->MaxValue << "\"];\n";
    }
}

std::string TreeVisualization::formatNodeXGBoost(const DecisionTreeXGBoost::Tree* node,
                                          const std::vector<std::string>& feature_names) {
    std::stringstream ss;

    if (node->IsLeaf) {
        ss << std::fixed << std::setprecision(6);
        ss << "Prediction: " << node->Prediction;
    } else {
        std::string feature_name = (node->FeatureIndex < feature_names.size())
            ? feature_names[node->FeatureIndex]
            : "f" + std::to_string(node->FeatureIndex);
        
        ss << feature_name << "\\n";
        ss << "Gain: " << std::scientific << std::setprecision(4) << node->GainImprovement << "\\n";
        ss << "Threshold: " << node->MaxValue << "\\n";
    }

    return ss.str();
}