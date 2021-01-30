#include "DotProductOperation.h"

#include <iostream>

#include <device_launch_parameters.h>
#include "cuda_util.h"


void DotProductFloatOperation::AllocateHost()
{
    h_A = (float*)malloc(elements * sizeof(float));
    h_B = (float*)malloc(elements * sizeof(float));
    h_C = (float*)malloc(threadsPerBlock * sizeof(float));

    if (h_C != nullptr)
    {
        h_C = (float*)memset(h_C, 0, threadsPerBlock * sizeof(float));
    }

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void DotProductFloatOperation::AllocateDevice()
{
    checkCudaError(cuMemAlloc(&d_A, elements * sizeof(float)));
    checkCudaError(cuMemAlloc(&d_B, elements * sizeof(float)));
    checkCudaError(cuMemAlloc(&d_C, threadsPerBlock * sizeof(float)));
    checkCudaError(cuMemsetD8(d_C, 0, threadsPerBlock * sizeof(float)));
}

void DotProductFloatOperation::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = (float) i;
    }
}

void DotProductFloatOperation::CopyToDevice()
{
    checkCudaError(cuMemcpyHtoD(d_A, h_A, elements * sizeof(float)));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, elements * sizeof(float)));
}

void DotProductFloatOperation::Launch()
{
    dim3 blockSize(threadsPerBlock);
    dim3 gridSize(1);

    void* args[5] = { &d_A, &d_B, &d_C, &elements };
    checkCudaError(cuLaunchKernel(GetFunction(),
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        threadsPerBlock * 2 * sizeof(float),
        NULL, args, NULL));
}

void DotProductFloatOperation::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, threadsPerBlock * sizeof(float)));

    for (int i = 1; i < threadsPerBlock; i++)
    {
        h_C[0] += h_C[i];
    }
}

void DotProductFloatOperation::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i] * h_B[i];
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
    }
}

void DotProductFloatOperation::FreeResources()
{
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}
