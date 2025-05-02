#include "../functions/math/math_functions.h"
#include "../pipeline/model_params.h"
#include "../pipeline/run_models.h"
#include "../pipeline/data_split.h"
#include <chrono>
#include <iomanip>
#include <memory>

int main(int argc, char *argv[]) {
   
    #ifdef USE_MPI 
    int already = 0;
    MPI_Initialized(&already);
    if (!already) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv,
                      MPI_THREAD_FUNNELED, /* requested level */
                      &provided);
        /* provided can be FUNNELED or MULTIPLE – both are fine */
    }
    #endif

    ProgramOptions programOptions = parseCommandLineArguments(argc, argv);
    DataParams data_params;
    auto it = programOptions.flags.find("data");
    if (it != programOptions.flags.end()) {
        data_params.dataPath = it->second;
    }
    if (!splitDataset(data_params)) {
        return -1;
    }
    
    switch (programOptions.choice) {
        case 1: {
            DecisionTreeParams treeParams;
            if (!getDecisionTreeParams(programOptions, treeParams)) return -1;
            runSingleDecisionTreeModel(treeParams, data_params);
            break;
        }
        case 2: {
            BaggingParams baggingParams;
            if (!getBaggingParams(programOptions, baggingParams)) return -1;
            runBaggingModel(baggingParams, data_params);
            break;
        }
        case 3: {
            BoostingParams boostingParams;
            if (!getBoostingParams(programOptions, boostingParams)) return -1;
            runBoostingModel(boostingParams, data_params);
            break;
        }
        case 4: {
            LightGBMParams lgbParams;
            if (!getLightGBMParams(programOptions, lgbParams)) return -1;
            runLightGBMModel(lgbParams, data_params);
            break;
        }
        case 5: { // ——Advanced GBDT / DART——
            AdvGBDTParams advParams;
            if (!getAdvGBDTParams(programOptions, advParams)) return -1;
            runAdvGBDTModel(advParams, data_params);
            break;
        }
        default:
            std::cerr << "Invalid choice! Please choose 1-5" << std::endl;
            return -1;
    }

    
    #ifdef USE_MPI
    MPI_Finalize();
    #endif

    return 0;
}