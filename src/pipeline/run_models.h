#pragma once

#include "../functions/feature/feature_importance.h"
#include "../functions/tree/vizualization/tree_visualization.h"
#include "../model_comparison/model_comparison.h"
#include "model_params.h"
#include "data_split.h"

// ===========================================
// Déclarations des fonctions d'exécution
// ===========================================

void runSingleDecisionTreeModel(DecisionTreeParams params, DataParams data_params);

void runBaggingModel(BaggingParams params, DataParams data_params);

void runBoostingModel(BoostingParams params, DataParams data_params);