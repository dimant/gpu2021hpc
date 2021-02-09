#include <iostream>

#include <device_launch_parameters.h>

#include "cuda_util.h"
#include "BlurCuda.h"

void BlurCuda::AllocateDevice()
{
    size_t size = rows * cols * sizeof(unsigned char);

    checkCudaError(cuMemAlloc(&d_A, size));
    checkCudaError(cuMemAlloc(&d_B, size));
}

void BlurCuda::CopyToDevice()
{
    size_t size = rows * cols * sizeof(char);

    checkCudaError(cuMemcpyHtoD(d_A, h_A, size));
}

void BlurCuda::Launch()
{
    dim3 blockSize(GetContext().work.threads.x, GetContext().work.threads.y);
    dim3 gridSize(GetContext().work.blocks.x, GetContext().work.blocks.y);

    void* args[5] = { &d_A, &d_B, &rows, &cols, &blurSize };

    checkCudaError(cuLaunchKernel(
        GetContext().cuFunction,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, args, NULL));
}

void BlurCuda::CopyFromDevice()
{
    size_t size = rows * cols * sizeof(unsigned char);

    checkCudaError(cuMemcpyDtoH(h_B, d_B, size));
}

void BlurCuda::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
}