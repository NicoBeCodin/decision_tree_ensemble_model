#ifndef GET_MODEL_PARAMETERS_H
#define GET_MODEL_PARAMETERS_H

#include "../model_comparison/model_comparison.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <thread>

void getModelParameters(int model_choice, std::string& parameters);

#endif // GET_MODEL_PARAMETERS_H