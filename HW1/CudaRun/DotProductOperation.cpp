#include "DotProductOperation.h"

#include <iostream>
#include <math.h>

#include <device_launch_parameters.h>
#include "cuda_util.h"

template class DotProductOperation<float>;
template class DotProductOperation<float2>;
template class DotProductOperation<float4>;

template <class T>
void DotProductOperation<T>::AllocateHost()
{
    h_A = (T*)malloc(elements * sizeof(T));
    h_B = (T*)malloc(elements * sizeof(T));
    h_C = (float*)malloc(blocksPerGrid * sizeof(float));

    if (h_C != nullptr)
    {
        h_C = (float*)memset(h_C, 0, blocksPerGrid * sizeof(float));
    }

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <class T>
void DotProductOperation<T>::AllocateDevice()
{
    checkCudaError(cuMemAlloc(&d_A, elements * sizeof(T)));
    checkCudaError(cuMemAlloc(&d_B, elements * sizeof(T)));
    checkCudaError(cuMemAlloc(&d_C, threadsPerBlock * sizeof(float)));
    checkCudaError(cuMemsetD8(d_C, 0, threadsPerBlock * sizeof(float)));
}

template <>
void DotProductOperation<float>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = log(float(i));
    }
}

template<>
void DotProductOperation<float2>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i].x = 1.0f;
        h_A[i].y = 1.0f;
        h_B[i].x = log(float(i));
        h_B[i].y = log(float(i));
    }
}

template<>
void DotProductOperation<float4>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i].x = 1.0f;
        h_A[i].y = 1.0f;
        h_A[i].z = 1.0f;
        h_A[i].w = 1.0f;
        h_B[i].x = log(float(i));
        h_B[i].y = log(float(i));
        h_B[i].z = log(float(i));
        h_B[i].w = log(float(i));
    }
}

template <class T>
void DotProductOperation<T>::CopyToDevice()
{
    checkCudaError(cuMemcpyHtoD(d_A, h_A, elements * sizeof(T)));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, elements * sizeof(T)));
}

template <class T>
void DotProductOperation<T>::Launch()
{
    dim3 blockSize(threadsPerBlock);
    dim3 gridSize(blocksPerGrid);

    int offset = 0;
    char argBuffer[256];

    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = d_A;
    offset += sizeof(d_A);
    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = d_B;
    offset += sizeof(d_B);
    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = d_C;
    offset += sizeof(d_C);

    *(reinterpret_cast<CUdeviceptr*>(&argBuffer[offset])) = elements;
    offset += sizeof(elements);

    void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     CU_LAUNCH_PARAM_END };

    checkCudaError(cuLaunchKernel(GetFunction(),
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        threadsPerBlock * 2 * sizeof(T),
        NULL, NULL, reinterpret_cast<void**>(&config)));
}

template <class T>
void DotProductOperation<T>::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, blocksPerGrid * sizeof(float)));

    for (int i = 1; i < blocksPerGrid; i++)
    {
        h_C[0] += h_C[i];
    }
}

template <>
void DotProductOperation<float>::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i] * h_B[i];
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed! %f != %f\n", c, h_C[0]);
        exit(EXIT_FAILURE);
    }
}

template <>
void DotProductOperation<float2>::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i].x * h_B[i].x;
        c += h_A[i].y * h_B[i].y;
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
    }
}

template <>
void DotProductOperation<float4>::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i].x * h_B[i].x;
        c += h_A[i].y * h_B[i].y;
        c += h_A[i].z * h_B[i].z;
        c += h_A[i].w * h_B[i].w;
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
    }
}

template <class T>
void DotProductOperation<T>::FreeHost()
{
    free(h_A);
    free(h_B);
    free(h_C);
}

template <class T>
void DotProductOperation<T>::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}
