#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../tree/math_functions.h"

namespace Visualization {

struct ModelComparison {
    std::string model_name;
    double mse;
    double mae;
    double r2_score;
    double training_time;
    double prediction_time;
};

class VisualizationUtils {
public:
    // Generates a comparison plot of metrics between different models
    static void generateModelComparisonPlot(const std::vector<ModelComparison>& comparisons, 
                                          const std::string& output_file);

    // Generates a learning curve plot
    static void generateLearningCurve(const std::vector<double>& training_errors,
                                    const std::vector<double>& validation_errors,
                                    const std::string& output_file);

    // Generates a feature importance plot
    static void generateFeatureImportancePlot(const std::vector<std::pair<std::string, double>>& importance_scores,
                                            const std::string& output_file);

    // Calculates the R² coefficient
    static double calculateR2Score(const std::vector<double>& y_true, 
                                 const std::vector<double>& y_pred);

    // Calculates the Mean Absolute Error (MAE)
    static double calculateMAE(const std::vector<double>& y_true, 
                             const std::vector<double>& y_pred);

private:
    static void writeGnuplotScript(const std::string& script_content, 
                                  const std::string& script_file);
};

} // namespace Visualization

#endif // VISUALIZATION_UTILS_H
