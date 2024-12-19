#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../functions_tree/math_functions.h"

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
    // Génère un graphique de comparaison des métriques entre différents modèles
    static void generateModelComparisonPlot(const std::vector<ModelComparison>& comparisons, 
                                          const std::string& output_file);

    // Génère une courbe d'apprentissage (learning curve)
    static void generateLearningCurve(const std::vector<double>& training_errors,
                                    const std::vector<double>& validation_errors,
                                    const std::string& output_file);

    // Génère un graphique de l'importance des caractéristiques
    static void generateFeatureImportancePlot(const std::vector<std::pair<std::string, double>>& importance_scores,
                                            const std::string& output_file);

    // Calcule le coefficient R²
    static double calculateR2Score(const std::vector<double>& y_true, 
                                 const std::vector<double>& y_pred);

    // Calcule l'erreur moyenne absolue (MAE)
    static double calculateMAE(const std::vector<double>& y_true, 
                             const std::vector<double>& y_pred);

private:
    static void writeGnuplotScript(const std::string& script_content, 
                                  const std::string& script_file);
};

} // namespace Visualization

#endif // VISUALIZATION_UTILS_H 