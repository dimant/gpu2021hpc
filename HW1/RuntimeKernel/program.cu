#include <stdio.h>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements);

inline void cudaCheckError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void allocateHostVectors(float** A, float** B, float** C, int numElements)
{
    size_t size = numElements * sizeof(float);

    *A = (float*)malloc(size);
    *B = (float*)malloc(size);
    *C = (float*)malloc(size);

    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "failed to allocate host vectors");
        exit(EXIT_FAILURE);
    }
}

void allocateDeviceVectors(float** A, float** B, float** C, int numElements)
{
    size_t size = numElements * sizeof(float);

    cudaCheckError(cudaMalloc(A, size));
    cudaCheckError(cudaMalloc(B, size));
    cudaCheckError(cudaMalloc(C, size));
}

int main(int argc, char** argv)
{
    float* h_A, * d_A;
    float* h_B, * d_B;
    float* h_C, * d_C;

    int numElements = 50000;

    allocateHostVectors(&h_A, &h_B, &h_C, numElements);

    for (int i = 0; i < numElements; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    allocateDeviceVectors(&d_A, &d_B, &d_C, numElements);

    size_t size = numElements * sizeof(float);

    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (d_A, d_B, d_C, numElements);

    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}