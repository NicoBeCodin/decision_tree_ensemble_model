#pragma once

#include "../functions/tree/decision_tree_single.h"
#include "../ensemble/bagging/bagging.h"
#include "../ensemble/boosting/boosting.h"
#include "../ensemble/boosting_lightgbm/my_lightgbm.h"
#include "../ensemble/boosting_advanced/boosting_advanced.h"
#include "../utils/utility.h"
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
    bool useSplitHistogram;
    bool useOMP;
    int numThreads;
    int mpiProcs;
};

struct BaggingParams {
    int numTrees;
    int maxDepth;
    int minSamplesSplit;
    double minImpurityDecrease;
    int criteria;
    int whichLossFunction;
    bool useSplitHistogram;
    bool useOMP;
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
    bool useSplitHistogram;
    bool useOMP;
    int numThreads;
};



struct LightGBMParams {
    int    nEstimators;     
    double learningRate;     
    int    maxDepth;         
    int    numLeaves;       
    double subsample;        
    double colsampleBytree;  
};
enum class AdvBinMethod { Quantile = 0, Frequency = 1 };

struct AdvGBDTParams {
    int    nEstimators;
    double learningRate;
    int    maxDepth;
    size_t minDataLeaf;
    int    numBins;
    bool   useDart;
    double dropoutRate;
    double skipDropRate;
    int    numThreads;
    AdvBinMethod binMethod;
};

// ===============================
// Fonctions de récupération des paramètres
// ===============================

bool getDecisionTreeParams(const ProgramOptions& options, DecisionTreeParams& out_params);

bool getBaggingParams(const ProgramOptions& options, BaggingParams& out_params);

bool getBoostingParams(const ProgramOptions& options, BoostingParams& out_params);

bool getLightGBMParams(const ProgramOptions& options, LightGBMParams& out_params);

bool getAdvGBDTParams(const ProgramOptions& opt, AdvGBDTParams& out);
