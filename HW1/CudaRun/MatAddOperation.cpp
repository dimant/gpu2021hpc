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
    unsigned int threads = 1;
    dim3 blockSize(threads, threads);
    dim3 gridSize(1, 1);

    void* args[5] = { &d_A, &d_B, &d_C, &nrows, &ncols };
    checkCudaError(cuLaunchKernel(GetFunction(),
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        threads * threads * sizeof(float),
        NULL, args, NULL));
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