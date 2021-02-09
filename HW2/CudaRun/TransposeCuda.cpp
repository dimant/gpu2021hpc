#include <iostream>

#include <device_launch_parameters.h>

#include "cuda_util.h"
#include "TransposeCuda.h"

void TransposeCuda::AllocateDevice()
{
    size_t size = rows * cols * sizeof(int);

    checkCudaError(cuMemAlloc(&d_A, size));
    checkCudaError(cuMemAlloc(&d_B, size));
}

void TransposeCuda::CopyToDevice()
{
    size_t size = rows * cols * sizeof(int);

    checkCudaError(cuMemcpyHtoD(d_A, h_A, size));
}

void TransposeCuda::Launch()
{
    dim3 blockSize(GetContext().work.threads.x, GetContext().work.threads.y);
    dim3 gridSize(GetContext().work.blocks.x, GetContext().work.blocks.y);

    void* args[4] = { &d_A, &d_B, &rows, &cols };

    checkCudaError(cuLaunchKernel(
        GetContext().cuFunction, 
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, args, NULL));
}

void TransposeCuda::CopyFromDevice()
{
    size_t size = rows * cols * sizeof(int);

    checkCudaError(cuMemcpyDtoH(h_B, d_B, size));
}

void TransposeCuda::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
}