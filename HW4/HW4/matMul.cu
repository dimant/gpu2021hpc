#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matMulGlobal(float* A, float* B, float* C, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < width && col < width)
	{
		float matmul = 0;

		for (int i = 0; i < width; ++i)
		{
			matmul += A[row * width + i] * B[i * width + col];
		}

		C[row * width + col] = matmul;
	}
}

__global__ void matMulShared(float* A, float* B, float* C, int width)
{
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * blockDim.y + ty;
	int col = blockIdx.x * blockDim.x + tx;
	
	float matmul = 0;

	// iterates over matrix in strides of size TILE_WIDTH
	// the iteration happens in x and y direction because the indices are dependent on tx and ty
	for (int p = 0; p < width / TILE_WIDTH; ++p)
	{
		// load all cells of the tile. each cell is loaded by a different thread
		ds_A[ty][tx] = A[row * width + p * TILE_WIDTH + tx];
		ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * width + col];

		__syncthreads();

		// calculate partial sum for tile.
		for (int i = 0; i < TILE_WIDTH; ++i)
		{
			matmul += ds_A[ty][i] * ds_B[i][tx];
		}

		__syncthreads();
	}

	C[row * width + col] = matmul;
}

