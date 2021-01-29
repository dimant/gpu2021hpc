#include <device_launch_parameters.h>

__global__ void matAdd(const float* A, const float* B, float* C, int nrows, int ncols)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < nrows && col < ncols)
    {
        int idx = row * ncols + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void matAddRow(const float* A, const float* B, float* C, int nrows, int ncols)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < nrows)
    {
        for (int col = 0; col < ncols; col++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void matAddCol(const float* A, const float* B, float* C, int nrows, int ncols)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < ncols)
    {
        for (int row = 0; row < nrows; row++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

