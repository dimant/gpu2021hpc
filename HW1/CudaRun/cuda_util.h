#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <string>
#include <cuda.h>

std::string readFile(const char* fileName);

// Sets the current working directory to be the same as the directory
// containing the running executable.
void setCwdToExeDir();

#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
void __checkCudaError(CUresult error, const char* file, const int line);

void cudaCompileKernel(
    CUdevice cuDevice,
    const char* kernelFile,
    char** cubinResult,
    size_t* cubinResultSize);

#endif