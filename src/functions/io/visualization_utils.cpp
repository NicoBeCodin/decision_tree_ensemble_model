#include "visualization_utils.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace Visualization {

void VisualizationUtils::generateModelComparisonPlot(
    const std::vector<ModelComparison>& comparisons, 
    const std::string& output_file) {
    
    std::ofstream data_file("model_comparison_data.txt");
    for (const auto& comp : comparisons) {
        data_file << comp.model_name << " " 
                 << comp.mse << " " 
                 << comp.mae << " " 
                 << comp.r2_score << " "
                 << comp.training_time << "\n";
    }
    data_file.close();

    std::stringstream script;
    script << "set terminal pngcairo enhanced font 'Arial,12' size 1200,800\n"
           << "set output '" << output_file << "'\n"
           << "set style data histograms\n"
           << "set style fill solid 1.0\n"
           << "set title 'Comparaison des Modèles'\n"
           << "set xlabel 'Modèles'\n"
           << "set ylabel 'Valeurs'\n"
           << "set key outside\n"
           << "set xtic rotate by -45\n"
           << "plot 'model_comparison_data.txt' using 2:xtic(1) title 'MSE', \\\n"
           << "     '' using 3 title 'MAE', \\\n"
           << "     '' using 4 title 'R²', \\\n"
           << "     '' using 5 title 'Temps d\\'entraînement (s)'\n";

    writeGnuplotScript(script.str(), "model_comparison.plt");
    std::system("gnuplot model_comparison.plt");
    std::remove("model_comparison_data.txt");
    std::remove("model_comparison.plt");
}

void VisualizationUtils::generateLearningCurve(
    const std::vector<double>& training_errors,
    const std::vector<double>& validation_errors,
    const std::string& output_file) {
    
    std::ofstream data_file("learning_curve_data.txt");
    for (size_t i = 0; i < training_errors.size(); ++i) {
        data_file << i << " " 
                 << training_errors[i] << " " 
                 << validation_errors[i] << "\n";
    }
    data_file.close();

    std::stringstream script;
    script << "set terminal pngcairo enhanced font 'Arial,12' size 1200,800\n"
           << "set output '" << output_file << "'\n"
           << "set title 'Courbe d\\'Apprentissage'\n"
           << "set xlabel 'Itérations'\n"
           << "set ylabel 'Erreur'\n"
           << "set grid\n"
           << "plot 'learning_curve_data.txt' using 1:2 with lines title 'Erreur d\\'entraînement', \\\n"
           << "     '' using 1:3 with lines title 'Erreur de validation'\n";

    writeGnuplotScript(script.str(), "learning_curve.plt");
    std::system("gnuplot learning_curve.plt");
    std::remove("learning_curve_data.txt");
    std::remove("learning_curve.plt");
}

void VisualizationUtils::generateFeatureImportancePlot(
    const std::vector<std::pair<std::string, double>>& importance_scores,
    const std::string& output_file) {
    
    std::ofstream data_file("feature_importance_data.txt");
    for (const auto& score : importance_scores) {
        data_file << score.first << " " << score.second << "\n";
    }
    data_file.close();

    std::stringstream script;
    script << "set terminal pngcairo enhanced font 'Arial,12' size 1200,800\n"
           << "set output '" << output_file << "'\n"
           << "set style data histogram\n"
           << "set style fill solid 1.0\n"
           << "set title 'Importance des Caractéristiques'\n"
           << "set xlabel 'Caractéristiques'\n"
           << "set ylabel 'Score d\\'importance'\n"
           << "set xtic rotate by -45\n"
           << "plot 'feature_importance_data.txt' using 2:xtic(1) title 'Score'\n";

    writeGnuplotScript(script.str(), "feature_importance.plt");
    std::system("gnuplot feature_importance.plt");
    std::remove("feature_importance_data.txt");
    std::remove("feature_importance.plt");
}

double VisualizationUtils::calculateR2Score(
    const std::vector<double>& y_true, 
    const std::vector<double>& y_pred) {
    
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return 0.0;
    }

    double y_mean = 0.0;
    for (const auto& y : y_true) {
        y_mean += y;
    }
    y_mean /= y_true.size();

    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_tot += (y_true[i] - y_mean) * (y_true[i] - y_mean);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }

    return 1.0 - (ss_res / ss_tot);
}

double VisualizationUtils::calculateMAE(
    const std::vector<double>& y_true, 
    const std::vector<double>& y_pred) {
    
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return 0.0;
    }

    double mae = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        mae += std::abs(y_true[i] - y_pred[i]);
    }
    return mae / y_true.size();
}

void VisualizationUtils::writeGnuplotScript(
    const std::string& script_content, 
    const std::string& script_file) {
    
    std::ofstream script_stream(script_file);
    script_stream << script_content;
    script_stream.close();
}

} // namespace Visualization 