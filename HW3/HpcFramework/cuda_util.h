#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda.h>

#define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
void __checkCudaError(CUresult error, const char* file, const int line);

void cudaCompileKernel(
    CUdevice cuDevice,
    CUlinkState linkState,
    const char* kernelFile,
    void** cubinResult,
    size_t* cubinResultSize,
    const char* cudadevrt = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib\\x64\\cudadevrt.lib");

void cudaMemoryTest();

#endif