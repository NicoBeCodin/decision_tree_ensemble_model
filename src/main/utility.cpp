#include "utility.h"


//For the single decision tree multithreading 
int adjustNumThreads(int numThreads) {
    if (numThreads <= 0) return 1;
    if ((numThreads & (numThreads - 1)) == 0) return numThreads;

    int power = 1;
    while (power * 2 <= numThreads) power *= 2;

    return power;
}
