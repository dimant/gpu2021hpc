#include <iostream>

#include <device_launch_parameters.h>

#include "cuda_util.h"
#include "MatAddCuda.h"

void MatAddCuda::AllocateDevice()
{
    size_t size = GetSize();

    checkCudaError(cuMemAlloc(&d_A, size));
    checkCudaError(cuMemAlloc(&d_B, size));
    checkCudaError(cuMemAlloc(&d_C, size));
}

void MatAddCuda::CopyToDevice()
{
    size_t size = GetSize();

    checkCudaError(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, size));
}

void MatAddCuda::Launch()
{
    unsigned int threads = 32;
    dim3 blockSize(threads, threads);
    dim3 gridSize(4, 4);

    int offset = 0;
    char argBuffer[256];

    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = d_A;
    offset += sizeof(d_A);
    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = d_B;
    offset += sizeof(d_B);
    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = d_C;
    offset += sizeof(d_C);

    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = nrows;
    offset += sizeof(nrows);
    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = ncols;
    offset += sizeof(ncols);

    void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     CU_LAUNCH_PARAM_END };

    checkCudaError(cuLaunchKernel(GetContext().cuFunction,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

void MatAddCuda::CopyFromDevice()
{
    size_t size = GetSize();

    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, size));
}

void MatAddCuda::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}