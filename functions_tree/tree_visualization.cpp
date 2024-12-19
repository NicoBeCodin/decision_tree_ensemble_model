#include "tree_visualization.h"
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <chrono>

// Constantes pour limiter la taille des visualisations
const int MAX_DEPTH_VISUALIZATION = 5;  // Profondeur maximale pour la visualisation
const int MAX_NODES_VISUALIZATION = 50;  // Nombre maximal de nœuds à afficher

void TreeVisualization::generateDotFile(const DecisionTreeSingle& tree,
                                        const std::string& filename,
                                        const std::vector<std::string>& feature_names) {
    try {
        if (!tree.getRoot()) {
            std::cout << "L'arbre est vide, impossible de générer la visualisation." << std::endl;
            return;
        }

        // Créer le dossier pour les fichiers DOT
        std::filesystem::path dotPath = "visualizations/dot";
        if (!std::filesystem::exists(dotPath)) {
            std::filesystem::create_directories(dotPath);
        }

        // Créer le dossier pour les images PNG
        std::filesystem::path pngPath = std::filesystem::current_path().parent_path() / "tree_visualizations";
        if (!std::filesystem::exists(pngPath)) {
            std::filesystem::create_directories(pngPath);
        }

        // Chemins complets pour les fichiers
        auto dot_file = std::filesystem::absolute(dotPath / (filename + ".dot"));
        auto png_file = std::filesystem::absolute(pngPath / (filename + ".png"));

        // Créer le fichier DOT
        std::ofstream out(dot_file);
        if (!out.is_open()) {
            std::cout << "Impossible d'ouvrir le fichier DOT pour écriture" << std::endl;
            return;
        }

        // En-tête du fichier DOT
        out << "digraph DecisionTree {\n";
        out << "    rankdir=TB;\n";
        out << "    node [shape=box, style=\"rounded,filled\", color=black, fontname=helvetica, fontsize=10];\n";
        out << "    edge [fontname=helvetica, fontsize=9];\n";
        out << "    graph [ranksep=0.3, nodesep=0.3];\n";

        // Générer le contenu de l'arbre
        int node_count = 0;
        int total_nodes = 0;
        generateDotContent(tree.getRoot(), out, node_count, feature_names, 0, total_nodes);

        out << "}\n";
        out.close();

        // Générer l'image PNG avec Graphviz
        std::stringstream cmd;
        std::string dot_path = getenv("DOT_PATH") ? getenv("DOT_PATH") : "dot";
        cmd << dot_path << " -Tpng -Gdpi=150 \"" << dot_file.string() << "\" -o \"" << png_file.string() << "\"";

        int result = system(cmd.str().c_str());
        if (result != 0) {
            std::cout << "Erreur lors de la génération du PNG (code: " << result << "). Assurez-vous que Graphviz est installé et que le chemin est correct." << std::endl;
        } else {
            std::cout << "Image générée: " << png_file.string() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Erreur: " << e.what() << std::endl;
    }
}

void TreeVisualization::generateDotContent(const DecisionTreeSingle::Tree* node,
                                         std::ofstream& out,
                                         int& node_count,
                                         const std::vector<std::string>& feature_names,
                                         int depth,
                                         int& total_nodes) {
    if (!node || depth >= MAX_DEPTH_VISUALIZATION || total_nodes >= MAX_NODES_VISUALIZATION) return;

    int current_node = node_count++;
    total_nodes++;
    
    std::string node_label = formatNode(node, feature_names);
    std::string node_color = node->IsLeaf ? "lightblue" : "lightgreen";
    out << "    node" << current_node << " [label=\"" << node_label << "\", fillcolor=" << node_color << "];\n";

    if (!node->IsLeaf && depth < MAX_DEPTH_VISUALIZATION - 1) {
        int left_child = node_count;
        generateDotContent(node->Left.get(), out, node_count, feature_names, depth + 1, total_nodes);
        out << "    node" << current_node << " -> node" << left_child 
            << " [label=\"≤" << std::fixed << std::setprecision(1) << node->MaxValue << "\"];\n";

        int right_child = node_count;
        generateDotContent(node->Right.get(), out, node_count, feature_names, depth + 1, total_nodes);
        out << "    node" << current_node << " -> node" << right_child 
            << " [label=\">" << std::fixed << std::setprecision(1) << node->MaxValue << "\"];\n";
    }
}

std::string TreeVisualization::formatNode(const DecisionTreeSingle::Tree* node,
                                        const std::vector<std::string>& feature_names) {
    std::stringstream ss;

    if (node->IsLeaf) {
        ss << std::fixed << std::setprecision(6);  // Plus de précision pour les feuilles
        ss << "Pred: " << node->Prediction;
    } else {
        std::string feature_name = (node->FeatureIndex < feature_names.size()) 
            ? feature_names[node->FeatureIndex] 
            : "f" + std::to_string(node->FeatureIndex);
        ss << feature_name << "\\n";
        ss << std::scientific << std::setprecision(4);  // Notation scientifique pour le MSE
        ss << "MSE: " << node->NodeMSE << "\\n";
        ss << "N: " << node->NodeSamples;
    }

    return ss.str();
}

void TreeVisualization::generateEnsembleDotFiles(
    const std::vector<std::unique_ptr<DecisionTreeSingle>>& trees,
    const std::string& base_filename,
    const std::vector<std::string>& feature_names) {
    
    std::cout << "Génération des arbres représentatifs..." << std::endl;
    
    // Indices des arbres à visualiser
    std::vector<size_t> important_indices;
    if (trees.size() > 0) {
        important_indices.push_back(0);  // Premier arbre
    }
    if (trees.size() > 2) {
        important_indices.push_back(trees.size() / 2);  // Arbre du milieu
    }
    if (trees.size() > 1) {
        important_indices.push_back(trees.size() - 1);  // Dernier arbre
    }

    for (size_t idx : important_indices) {
        std::string suffix;
        if (idx == 0) suffix = "_first";
        else if (idx == trees.size() - 1) suffix = "_last";
        else suffix = "_middle";

        std::string filename = base_filename + suffix;
        std::cout << "\nGénération de l'arbre " << (idx + 1) << "/" << trees.size() 
                  << " (position: " << suffix.substr(1) << ")" << std::endl;
        generateDotFile(*trees[idx], filename, feature_names);
    }

    std::cout << "Visualisations générées dans le dossier 'visualizations'" << std::endl;
} 