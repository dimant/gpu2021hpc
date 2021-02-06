#include "DotProductCuda.h"

#include <iostream>

#include "cuda_util.h"

template class DotProductCuda<float>;
template class DotProductCuda<float2>;
template class DotProductCuda<float4>;

template <class T>
void DotProductCuda<T>::AllocateHost()
{
    DotProductOperation<T>::AllocateHost();
}

template <class T>
void DotProductCuda<T>::InitData()
{
    DotProductOperation<T>::InitData();
}

template <class T>
void DotProductCuda<T>::FreeHost()
{
    DotProductOperation<T>::FreeHost();
}

template <class T>
void DotProductCuda<T>::VerifyResult()
{
    DotProductOperation<T>::VerifyResult();
}

template <class T>
void DotProductCuda<T>::AllocateDevice()
{
    checkCudaError(cuMemAlloc(&d_A, elements * sizeof(T)));
    checkCudaError(cuMemAlloc(&d_B, elements * sizeof(T)));
    checkCudaError(cuMemAlloc(&d_C, blocksPerGrid * sizeof(float)));
    checkCudaError(cuMemsetD8(d_C, 0, blocksPerGrid * sizeof(float)));
}

template <class T>
void DotProductCuda<T>::CopyToDevice()
{
    checkCudaError(cuMemcpyHtoD(d_A, h_A, elements * sizeof(T)));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, elements * sizeof(T)));
}

template <class T>
void DotProductCuda<T>::Launch()
{
    dim3 blockSize(threadsPerBlock);
    dim3 gridSize(blocksPerGrid);

    int offset = 0;
    char argBuffer[1024];

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_A), sizeof(d_A));
    offset += sizeof(d_A);

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_B), sizeof(d_B));
    offset += sizeof(d_B);

    ALIGN_UP(offset, __alignof(CUdeviceptr));
    memcpy(argBuffer + offset, &(d_C), sizeof(d_C));
    offset += sizeof(d_C);

    ALIGN_UP(offset, __alignof(size_t));
    memcpy(argBuffer + offset, &(elements), sizeof(elements));
    offset += sizeof(elements);

    void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     CU_LAUNCH_PARAM_END };

    checkCudaError(cuLaunchKernel(GetContext().cuFunction,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        threadsPerBlock * 2 * sizeof(T),
        NULL, NULL, reinterpret_cast<void**>(&config)));
}

template <class T>
void DotProductCuda<T>::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, blocksPerGrid * sizeof(float)));

    for (int i = 1; i < blocksPerGrid; i++)
    {
        h_C[0] += h_C[i];
    }
}

template <class T>
void DotProductCuda<T>::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}
