#include "cuda_util.h"

#include <iostream>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

#include "util.h"

#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
void __checkCudaError(CUresult error, const char* file, const int line)
{
    if (error != CUDA_SUCCESS)
    {
        const char* errorStr = NULL;
        cuGetErrorString(error, &errorStr);

        fprintf(stderr, "%s:%d Error: %s\n", file, line, errorStr);
        exit(EXIT_FAILURE);
    }
}

#define checkNvrtcError(err) __checkNvrtcError(err, __FILE__, __LINE__)
void __checkNvrtcError(nvrtcResult error, const char* file, const int line)
{
    if (error != NVRTC_SUCCESS)
    {
        fprintf(stderr, "%s:%d Error: %s\n", file, line, nvrtcGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void cudaCompileKernel(
    CUdevice cuDevice,
    CUlinkState linkState,
    const char* kernelFile,
    void** cubinResult,
    size_t* cubinResultSize,
    const char* cudadevrt)
{
    std::string kernelSource = readFile(kernelFile);
    const char* kernelString = kernelSource.c_str();

    const char* compileParams[256];
    int nCompileParams = 0;

    int major = 0, minor = 0;

    checkCudaError(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    checkCudaError(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    std::string optionGpuArchitecture = std::string("--gpu-architecture=sm_")
        + std::to_string(major)
        + std::to_string(minor);

#ifdef _DEBUG
    compileParams[nCompileParams++] = "--device-debug";
#else

#endif

    compileParams[nCompileParams++] = "--fmad=false";
    compileParams[nCompileParams++] = optionGpuArchitecture.c_str();
    compileParams[nCompileParams++] = "--relocatable-device-code=true";

    nvrtcProgram program;

    std::cerr << std::endl << "Compiling: " << kernelFile << std::endl;

    checkNvrtcError(nvrtcCreateProgram(&program, kernelSource.c_str(), kernelFile, 0, NULL, NULL));
    nvrtcResult compileError = nvrtcCompileProgram(program, nCompileParams, compileParams);
    
    size_t logSize;
    checkNvrtcError(nvrtcGetProgramLogSize(program, &logSize));
    char* log = new char[logSize + 1];
    checkNvrtcError(nvrtcGetProgramLog(program, log));
    log[logSize] = '\0';

    if (strlen(log) >= 2)
    {
        std::cerr << log;
    }
    else
    {
        std::cerr << " done." << std::endl;
    }

    delete[] log;

    checkNvrtcError(compileError);

    std::cerr << std::endl << "Linking: " << kernelFile << std::endl;
    
    size_t ptxSize;
    checkNvrtcError(nvrtcGetPTXSize(program, &ptxSize));
    char* ptx = new char[ptxSize];
    checkNvrtcError(nvrtcGetPTX(program, ptx));
    // Destroy the program.
    checkNvrtcError(nvrtcDestroyProgram(&program));

    checkCudaError(cuLinkCreate(0, 0, 0, &linkState));
    checkCudaError(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, cudadevrt, 0, 0, 0));
    checkCudaError(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx, ptxSize, kernelFile, 0, 0, 0));
    
    size_t cubinSize;
    void* cubin;
    checkCudaError(cuLinkComplete(linkState, &cubin, &cubinSize));

    std::cerr << " done." << std::endl;

    *cubinResult = cubin;
    *cubinResultSize = cubinSize;
}

void cudaMemoryTest()
{
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int* h_a = (int*)malloc(bytes);
    CUdeviceptr d_a;
    checkCudaError(cuMemAlloc(&d_a, bytes));

    if (h_a == nullptr)
    {
        std::cerr << "couldn't allocate h_a" << std::endl;
        exit(1);
    }
    else
    {
        memset(h_a, 0, bytes);
    }

    checkCudaError(cuMemcpyHtoD(d_a, h_a, bytes));
    checkCudaError(cuMemcpyDtoH(h_a, d_a, bytes));
}