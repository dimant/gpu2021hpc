#include <iostream>

#include <device_launch_parameters.h>

#include "cuda_util.h"
#include "MatAddOperation.h"

void MatAddOperation::AllocateHost()
{
    size_t size = GetSize();

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MatAddOperation::AllocateDevice()
{
    size_t size = GetSize();

    checkCudaError(cuMemAlloc(&d_A, size));
    checkCudaError(cuMemAlloc(&d_B, size));
    checkCudaError(cuMemAlloc(&d_C, size));
}

void MatAddOperation::InitData()
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            h_A[i * ncols + j] = i;
            h_B[i * ncols + j] = j;
        }
    }
}

void MatAddOperation::CopyToDevice()
{
    size_t size = GetSize();

    checkCudaError(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, size));
}

void MatAddOperation::Launch()
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

    checkCudaError(cuLaunchKernel(GetFunction(),
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

void MatAddOperation::CopyFromDevice()
{
    size_t size = GetSize();

    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, size));
}

void MatAddOperation::VerifyResult()
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            double a = h_A[i * ncols + j];
            double b = h_B[i * ncols + j];
            double c = h_C[i * ncols + j];
            if (fabs(a + b - c) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d,%d!\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
}

void MatAddOperation::FreeResources()
{
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}