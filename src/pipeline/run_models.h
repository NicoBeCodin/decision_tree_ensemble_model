#pragma once

#include "../functions/feature/feature_importance.h"
#include "../functions/tree/vizualization/tree_visualization.h"
#include "../model_comparison/model_comparison.h"
#include "model_params.h"
#include "data_split.h"
#include <memory>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// ===========================================
// Déclarations des fonctions d'exécution
// ===========================================

void runSingleDecisionTreeModel(DecisionTreeParams params, DataParams data_params);

void runBaggingModel(BaggingParams params, DataParams data_params);

void runBoostingModel(BoostingParams params, DataParams data_params);

void runLightGBMModel(const LightGBMParams& params, const DataParams& data_params);

void runAdvGBDTModel(const AdvGBDTParams& p, const DataParams& data);