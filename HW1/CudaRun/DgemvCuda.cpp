#include "DgemvCuda.h"

#include <iostream>

#include <device_launch_parameters.h>
#include "cuda_util.h"

void DgemvCuda::AllocateDevice()
{
    checkCudaError(cuMemAlloc(&d_A, rows * cols * sizeof(double)));
    checkCudaError(cuMemAlloc(&d_x, cols * sizeof(double)));
    checkCudaError(cuMemAlloc(&d_y, rows * sizeof(double)));
    checkCudaError(cuMemAlloc(&d_z, rows * sizeof(double)));
    checkCudaError(cuMemsetD8(d_z, 0, rows * sizeof(double)));
}

void DgemvCuda::CopyToDevice()
{
    checkCudaError(cuMemcpyHtoD(d_A, h_A, rows * cols * sizeof(double)));
    checkCudaError(cuMemcpyHtoD(d_x, h_x, cols * sizeof(double)));
    checkCudaError(cuMemcpyHtoD(d_y, h_y, rows * sizeof(double)));
}

void DgemvCuda::Launch()
{
    unsigned int threads = 256;
    dim3 blockSize(threads);
    dim3 gridSize(4);

    int offset = 0;
    char argBuffer[1024];

    ALIGN_UP(offset, __alignof(double));
    memcpy(argBuffer + offset, &(alpha), sizeof(alpha));
    offset += sizeof(alpha);

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_A), sizeof(d_A));
    offset += sizeof(d_A);

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_x), sizeof(d_x));
    offset += sizeof(d_x);

    ALIGN_UP(offset, __alignof(double));
    memcpy(argBuffer + offset, &(beta), sizeof(beta));
    offset += sizeof(beta);

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_y), sizeof(d_y));
    offset += sizeof(d_y);

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_z), sizeof(d_z));
    offset += sizeof(d_z);

    ALIGN_UP(offset, __alignof(size_t));
    memcpy(argBuffer + offset, &(rows), sizeof(rows));
    offset += sizeof(rows);

    ALIGN_UP(offset, __alignof(size_t));
    memcpy(argBuffer + offset, &(cols), sizeof(cols));
    offset += sizeof(cols);

    void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     CU_LAUNCH_PARAM_END };

    checkCudaError(cuLaunchKernel(GetContext().cuFunction,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

void DgemvCuda::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_z), d_z, rows * sizeof(double)));
}

void DgemvCuda::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_x));
    checkCudaError(cuMemFree(d_y));
    checkCudaError(cuMemFree(d_z));
}

