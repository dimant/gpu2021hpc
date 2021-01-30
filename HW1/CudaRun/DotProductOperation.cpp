#include "DotProductOperation.h"

#include <iostream>

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
    h_C = (T*)malloc(threadsPerBlock * sizeof(T));

    if (h_C != nullptr)
    {
        h_C = (T*)memset(h_C, 0, threadsPerBlock * sizeof(T));
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
    checkCudaError(cuMemAlloc(&d_C, threadsPerBlock * sizeof(T)));
    checkCudaError(cuMemsetD8(d_C, 0, threadsPerBlock * sizeof(T)));
}

template <>
void DotProductOperation<float>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = (float) i;
    }
}

template<>
void DotProductOperation<float2>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i].x = 1.0f;
        h_A[i].y = 1.0f;
        h_B[i].x = (float)i;
        h_B[i].y = (float)i;
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
        h_B[i].x = (float)i;
        h_B[i].y = (float)i;
        h_B[i].z = (float)i;
        h_B[i].w = (float)i;
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
    dim3 gridSize(1);

    void* args[5] = { &d_A, &d_B, &d_C, &elements };
    checkCudaError(cuLaunchKernel(GetFunction(),
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        threadsPerBlock * 2 * sizeof(T),
        NULL, args, NULL));
}

template <class T>
void DotProductOperation<T>::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, threadsPerBlock * sizeof(T)));

    for (int i = 1; i < threadsPerBlock; i++)
    {
        h_C[0] += h_C[i];
    }
}

template <class T>
void DotProductOperation<T>::VerifyResult()
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

template <class T>
void DotProductOperation<T>::FreeResources()
{
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}
