#include "GemmCuda.h"

#include <iostream>

#include <device_launch_parameters.h>
#include "cuda_util.h"

template class GemmCuda<float>;
template class GemmCuda<double>;

template <class T>
void GemmCuda<T>::AllocateDevice()
{
    checkCudaError(cuMemAlloc(&d_A, (size_t)widthA * (size_t)heightA * sizeof(T)));
    checkCudaError(cuMemAlloc(&d_B, (size_t)widthB * (size_t)heightB * sizeof(T)));
    checkCudaError(cuMemAlloc(&d_C, (size_t)widthB * (size_t)heightA * sizeof(T)));
}

template <class T>
void GemmCuda<T>::CopyToDevice()
{
    checkCudaError(cuMemcpyHtoD(d_A, h_A, (size_t)widthA * (size_t)heightA * sizeof(T)));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, (size_t)widthB * (size_t)heightB * sizeof(T)));
}

template <class T>
void GemmCuda<T>::Launch()
{
    unsigned int threads = 32;
    dim3 blockSize(threads, threads);
    dim3 gridSize(4, 4);

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

    ALIGN_UP(offset, __alignof(int));
    memcpy(argBuffer + offset, &(widthA), sizeof(widthA));
    offset += sizeof(widthA);

    ALIGN_UP(offset, __alignof(int));
    memcpy(argBuffer + offset, &(widthB), sizeof(widthB));
    offset += sizeof(widthB);

    void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     CU_LAUNCH_PARAM_END };

    checkCudaError(cuLaunchKernel(GetContext().cuFunction,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

template <class T>
void GemmCuda<T>::CopyFromDevice()
{
    checkCudaError(cuMemcpyDtoH(h_C, d_C, (size_t)widthB * (size_t)heightA * sizeof(T)));
}

template <class T>
void GemmCuda<T>::FreeDevice()
{
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
}

