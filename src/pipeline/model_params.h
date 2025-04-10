#pragma once

#include "../functions/tree/decision_tree_single.h"
#include "../ensemble/bagging/bagging.h"
#include "../ensemble/boosting/boosting.h"
#include "../main/utility.h"
#include <string>
#include <iostream>
#include <filesystem>

// ===============================
// Structures de paramètres
// ===============================

struct DecisionTreeParams {
    int maxDepth;
    int minSamplesSplit;
    double minImpurityDecrease;
    int criteria;
    int numThreads;
    int useOmp;
};

struct BaggingParams {
    int numTrees;
    int maxDepth;
    int minSamplesSplit;
    double minImpurityDecrease;
    int criteria;
    int whichLossFunction;
    int numThreads;
};

struct BoostingParams {
    int nEstimators;
    double learningRate;
    int maxDepth;
    int minSamplesSplit;
    double minImpurityDecrease;
    int criteria;
    int whichLossFunction;
};

// ===============================
// Fonctions de récupération des paramètres
// ===============================

bool getDecisionTreeParams(const ProgramOptions& options, DecisionTreeParams& out_params);

bool getBaggingParams(const ProgramOptions& options, BaggingParams& out_params);

bool getBoostingParams(const ProgramOptions& options, BoostingParams& out_params);