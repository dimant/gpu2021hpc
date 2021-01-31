#include "CudaModule.h"

#include <iostream>

#include "util.h"

CudaContext CudaModule::GetContext(const char* kernelName)
{
    checkCudaError(cuModuleGetFunction(&cuFunction, cuModule, kernelName));

    CudaContext result;

    result.cuFunction = cuFunction;

    return result;
}

void CudaModule::Init()
{
    int deviceCount = 0;

    checkCudaError(cuInit(0));

    checkCudaError(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    checkCudaError(cuDeviceGet(&cuDevice, 0));

    checkCudaError(cuCtxCreate(&cuContext, 0, cuDevice));
}

void CudaModule::Compile(const char* kernelFile)
{
    setCwdToExeDir();

    char* cubinResult;
    size_t cubinResultSize;

    cudaCompileKernel(cuDevice, kernelFile, &cubinResult, &cubinResultSize);

    checkCudaError(cuModuleLoadData(&cuModule, cubinResult));

    delete[] cubinResult;
}