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

__global__ void matAdd(const float* A, const float* B, float* C, int nrows, int ncols);

__global__ void matAddRow(const float* A, const float* B, float* C, int nrows, int ncols);

__global__ void matAddCol(const float* A, const float* B, float* C, int nrows, int ncols);

inline void cudaCheckError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void allocateHostVectors(float* &A, float* &B, float* &C, int nrows, int ncols)
{
    size_t size = nrows * ncols * sizeof(float);

    A = (float*) malloc(size);
    B = (float*) malloc(size);
    C = (float*) malloc(size);

    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "failed to allocate host vectors");
        exit(EXIT_FAILURE);
    }
}

void allocateDeviceVectors(float** A, float** B, float** C, int nrows, int ncols)
{
    size_t size = nrows * ncols * sizeof(float);

    cudaCheckError(cudaMalloc(A, size));
    cudaCheckError(cudaMalloc(B, size));
    cudaCheckError(cudaMalloc(C, size));
}

int main(int argc, char** argv)
{
    float* h_A, * d_A;
    float* h_B, * d_B;
    float* h_C, * d_C;

    int nrows = 10;
    int ncols = 10;

    allocateHostVectors(h_A, h_B, h_C, nrows, ncols);

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            h_A[i * ncols + j] = rand() / (float)RAND_MAX;
            h_B[i * ncols + j] = rand() / (float)RAND_MAX;
        }
    }

    allocateDeviceVectors(&d_A, &d_B, &d_C, nrows, ncols);

    size_t size = nrows * ncols * sizeof(float);

    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 blockSize(3, 3);
    int bx = (ncols + blockSize.x - 1) / blockSize.x;
    int by = (nrows + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);

    matAdd CUDA_KERNEL(gridSize, blockSize) ((const float*) d_A, (const float*) d_B, d_C, nrows, ncols);

    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; j++)
        {
            if (fabs(h_A[i * ncols + j] + h_B[i * ncols + j] - h_C[i * ncols + j]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
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