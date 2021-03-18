#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>

void cudaCheckError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
