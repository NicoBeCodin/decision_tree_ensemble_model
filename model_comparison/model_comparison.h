#ifndef MODEL_COMPARISON_H
#define MODEL_COMPARISON_H

#include <string>
#include <vector>
#include <map>

struct ModelResults {
    std::string model_name;
    double mse;
    double training_time;
    double evaluation_time;
    std::map<std::string, double> feature_importance;
    std::map<std::string, double> parameters;
};

class ModelComparison {
public:
    static void saveResults(const ModelResults& results);
    static void displayComparison();
    static void generateComparisonTable();
    static std::string getResultsPath() { return "../results/all_models_comparison.md"; }

private:
    static void createResultsDirectory();
};

#endif // MODEL_COMPARISON_H 