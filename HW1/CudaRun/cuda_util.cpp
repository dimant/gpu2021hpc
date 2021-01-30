#include "cuda_util.h"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

#include <windows.h>

#include <cuda.h>
#include <nvrtc.h>

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

std::string readFile(const char* fileName)
{
    std::ifstream t(fileName);
    std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

void setCwdToExeDir()
{
    HMODULE hMod = GetModuleHandle(NULL);
    char path[MAX_PATH];
    GetModuleFileNameA(hMod, path, MAX_PATH);

    // Find the last '\' or '/' and terminate the path there; it is now
    // the directory containing the executable.
    size_t i;
    for (i = strlen(path) - 1; i > 0 && path[i] != '/' && path[i] != '\\'; --i);
    path[i] = '\0';

    SetCurrentDirectoryA(path);
}

void cudaCompileKernel(
    CUdevice cuDevice,
    const char* kernelFile,
    char** cubinResult,
    size_t* cubinResultSize)
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

    compileParams[0] = optionGpuArchitecture.c_str();
    nCompileParams++;

    nvrtcProgram program;

    std::cerr << std::endl << "Compiling: " << kernelFile;

    checkNvrtcError(nvrtcCreateProgram(&program, kernelSource.c_str(), kernelFile, 0, NULL, NULL));
    nvrtcResult compileError = nvrtcCompileProgram(program, nCompileParams, compileParams);
    
    size_t logSize;
    checkNvrtcError(nvrtcGetProgramLogSize(program, &logSize));
    char* log = new char[logSize + 1];
    checkNvrtcError(nvrtcGetProgramLog(program, log));
    log[logSize] = '\0';

    if (strlen(log) >= 2) {
        std::cerr << log;
    }
    else
    {
        std::cerr << " done." << std::endl;
    }

    delete[] log;

    checkNvrtcError(compileError);

    size_t codeSize;
    checkNvrtcError(nvrtcGetCUBINSize(program, &codeSize));

    char* code = new char[codeSize];
    checkNvrtcError(nvrtcGetCUBIN(program, code));

    *cubinResult = code;
    *cubinResultSize = codeSize;
}