#include "DgemvOperation.h"

#include <iostream>

#include <device_launch_parameters.h>
#include "cuda_util.h"

void DgemvOperation::AllocateHost()
{
    h_A = (double*)malloc(cols * rows * sizeof(double));
    h_x = (double*)malloc(cols * sizeof(double));
    h_y = (double*)malloc(rows * sizeof(double));
    h_z = (double*)malloc(rows * sizeof(double));

    if (h_A == nullptr || h_x == nullptr || h_y == nullptr || h_z == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }

    h_z = (double*)memset(h_z, 0, rows * sizeof(double));
}

void DgemvOperation::AllocateDevice()
{
    checkCudaError(cuMemAlloc(&d_A, rows * cols * sizeof(double)));
    checkCudaError(cuMemAlloc(&d_x, cols * sizeof(double)));
    checkCudaError(cuMemAlloc(&d_y, rows * sizeof(double)));
    checkCudaError(cuMemAlloc(&d_z, rows * sizeof(double)));
    checkCudaError(cuMemsetD8(d_z, 0, rows * sizeof(double)));
}

void DgemvOperation::InitData()
{
    alpha = 2.0;
    beta = 4.0;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int idx = row * cols + col;
            h_A[idx] = idx;
        }
    }

    for (int i = 0; i < cols; i++)
    {
        h_x[i] = i;
    }

    for (int i = 0; i < rows; i++)
    {
        h_y[i] = i;
    }
}

void DgemvOperation::CopyToDevice()
{
    checkCudaError(cuMemcpyHtoD(d_A, h_A, rows * cols * sizeof(double)));
    checkCudaError(cuMemcpyHtoD(d_x, h_x, cols * sizeof(double)));
    checkCudaError(cuMemcpyHtoD(d_y, h_y, rows * sizeof(double)));
}

void DgemvOperation::Launch()
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

    checkCudaError(cuLaunchKernel(GetFunction(),
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

void DgemvOperation::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_z), d_z, rows * sizeof(double)));
}

void DgemvOperation::VerifyResult()
{
    double* z = new double[rows];

    double buf = 0.0;

    for (int row = 0; row < rows; row++)
    {
        buf = 0.0f;

        for (int col = 0; col < cols; col++)
        {
            buf += alpha * h_A[row * cols + col] * h_x[col];
        }

        z[row] = buf + beta * h_y[row];
    }

    for (int i = 0; i < rows; i++)
    {
        if (fabs(h_z[i] - z[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed!\n");
            exit(EXIT_FAILURE);
        }
    }

    delete[] z;
}

void DgemvOperation::FreeHost()
{
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_z);
}

void DgemvOperation::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_x));
    checkCudaError(cuMemFree(d_y));
    checkCudaError(cuMemFree(d_z));
}

