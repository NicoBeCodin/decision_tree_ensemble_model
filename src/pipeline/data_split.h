#pragma once

#include "../functions/io/functions_io.h"  // pour DataIO
#include "../utils/utility.h"              // pour createDirectory()

// ==============================
// Structure pour les datasets
// ==============================

struct DataParams {
    std::vector<double> X_train;
    std::vector<double> y_train;
    std::vector<double> X_test; 
    std::vector<double> y_test;
    int rowLength;
    std::string dataPath = "../datasets/processed/cleaned_data.csv"; // par défaut
};

// ==============================
// Fonction pour charger + découper les données
// ==============================

bool splitDataset(DataParams& data_params);