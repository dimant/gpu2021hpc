#pragma once

#include "cuda_runtime.h"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

inline void cudaCheckError(cudaError_t error);

