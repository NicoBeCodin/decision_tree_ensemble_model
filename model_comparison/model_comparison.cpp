#include "model_comparison.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

void ModelComparison::createResultsDirectory() {
    std::filesystem::path results_dir = "../results";
    if (!std::filesystem::exists(results_dir)) {
        std::filesystem::create_directories(results_dir);
    }
}

void ModelComparison::saveResults(const ModelResults& results) {
    createResultsDirectory();
    
    std::ofstream file(getResultsPath(), std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open results file");
    }

    // If file is empty, add header
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "# Decision Tree Models Comparison\n\n";
        file << "## Performance Results\n\n";
    }

    // Add model section
    file << "### " << results.model_name << "\n\n";
    
    // Performance metrics
    file << "#### Performance Metrics\n";
    file << "- MSE: " << std::scientific << std::setprecision(3) << results.mse << "\n";
    file << "- Training Time: " << std::fixed << std::setprecision(3) << results.training_time << " seconds\n";
    file << "- Evaluation Time: " << results.evaluation_time << " seconds\n\n";

    // Model parameters
    file << "#### Model Parameters\n";
    for (const auto& [param_name, param_value] : results.parameters) {
        file << "- " << param_name << ": " << param_value << "\n";
    }
    file << "\n";

    // Feature importance
    file << "#### Feature Importance\n";
    if (!results.feature_importance.empty()) {
        for (const auto& [feature, importance] : results.feature_importance) {
            file << "- " << feature << ": " << std::fixed << std::setprecision(2) 
                 << importance * 100.0 << "%\n";
        }
    } else {
        file << "Feature importance not available for this model.\n";
    }
    file << "\n---\n\n";

    file.close();
}

void ModelComparison::displayComparison() {
    std::ifstream file(getResultsPath());
    if (!file.is_open()) {
        std::cout << "No comparison results available.\n";
        std::cout << "Please run some models first.\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }
    file.close();
}

void ModelComparison::generateComparisonTable() {
    std::ifstream file(getResultsPath());
    if (!file.is_open()) {
        std::cout << "No comparison results available.\n";
        return;
    }

    std::cout << "\n=== Models Performance Comparison ===\n\n";
    
    // Read and display each model's results
    std::string line;
    bool in_model_section = false;
    std::string current_model;
    
    while (std::getline(file, line)) {
        if (line.find("### ") == 0) {
            current_model = line.substr(4);
            in_model_section = true;
            std::cout << "\n" << current_model << ":\n";
            continue;
        }
        
        if (in_model_section) {
            if (line.find("MSE:") != std::string::npos ||
                line.find("Training Time:") != std::string::npos ||
                line.find("Evaluation Time:") != std::string::npos) {
                std::cout << "  " << line << "\n";
            }
            
            if (line == "---") {
                in_model_section = false;
            }
        }
    }
    file.close();
} 