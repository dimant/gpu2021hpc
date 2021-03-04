#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include "popl.h"
#include "cudart_util.h"

__global__ void matMulGlobal(float* A, float* B, float* C, int width);
__global__ void matMulShared(float* A, float* B, float* C, int width);


void initMatrices(float* A, float* B, float* C, int width)
{
	int size = width * width;

	for (int i = 0; i < size; i++)
	{
		A[i] = (float)rand() / (float)(RAND_MAX);
		B[i] = (float)rand() / (float)(RAND_MAX);
		C[i] = 0.0;
	}
}

void allocateMatrices(float** A, float** B, float** C, int width)
{
	int size = width * width;

	*A = (float*)malloc(size * sizeof(float));
	*B = (float*)malloc(size * sizeof(float));
	*C = (float*)malloc(size * sizeof(float));

	cudaMallocManaged(&(*A), size * sizeof(float));
	cudaMallocManaged(&(*B), size * sizeof(float));
	cudaMallocManaged(&(*C), size * sizeof(float));
}

void freeMatrices(float* A, float* B, float* C)
{
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}

void matMulVerify(float* A, float* B, float* C, int width)
{
	int size = width * width;

	float* R = (float*)malloc(size * sizeof(float));
	float buffer = 0;

	for (int row = 0; row < width; row++)
	{
		for (int col = 0; col < width; col++)
		{
			buffer = 0;

			for (int i = 0; i < width; i++)
			{
				buffer += A[row * width + i] * B[i * width + col];
			}

			R[row * width + col] = buffer;
		}
	}

	float maxError = 0.f;

	for (int i = 0; i < size; i++)
	{
		float error = fabs(R[i] - C[i]);

		if (maxError < error)
		{
			maxError = error;
		}

		//if (error > 1e-5)
		//{
		//	fprintf(stderr, "Result verification failed at element %d!\n", i);
		//	//exit(EXIT_FAILURE);
		//}
	}

	printf("Max error: %f", maxError);
}

void matMulGlobalLaunch(int threads, int blocks, int width)
{
	float* A;
	float* B;
	float* C;

	allocateMatrices(&A, &B, &C, width);
	initMatrices(A, B, C, width);

	dim3 blockSize(threads, threads);
	dim3 gridSize(blocks, blocks);

	printf("Testing global memory matrix multiplication:\n");
	matMulGlobal CUDA_KERNEL(gridSize, blockSize) (A, B, C, width);
	cudaDeviceSynchronize();
	matMulVerify(A, B, C, width);
	freeMatrices(A, B, C);
}

void matMulSharedLaunch(int threads, int blocks, int width)
{
	float* A;
	float* B;
	float* C;

	allocateMatrices(&A, &B, &C, width);
	initMatrices(A, B, C, width);

	dim3 blockSize(threads, threads);
	dim3 gridSize(blocks, blocks);

	printf("Testing shared memory matrix multiplication:\n");
	matMulShared CUDA_KERNEL(gridSize, blockSize) (A, B, C, width);
	cudaDeviceSynchronize();
	matMulVerify(A, B, C, width);
	freeMatrices(A, B, C);
}

int main(int argc, char** argv)
{
	popl::OptionParser op("HW4 options");
	int threads, blocks, width;

	auto help_option = op.add<popl::Switch>("h", "help", "produce help message");
	auto threads_option = op.add<popl::Value<int>>("t", "threads", "Number of x and y threads", 16, &threads);
	auto blocks_option = op.add<popl::Value<int>>("b", "blocks", "Number of x and y blocks", 16, &blocks);
	auto width_option = op.add<popl::Value<int>>("w", "width", "Matrix width.", 256, &width);
	auto kernelName_option = op.add<popl::Value<std::string>>("k", "kernel-name", "Name of kernel.");

	op.parse(argc, argv);

	if (!kernelName_option->is_set() || help_option->is_set())
	{
		std::cout << op << "\n";
	}
	else
	{
		std::string kernelName = kernelName_option->value();

		if (kernelName.rfind("matMulGlobal", 0) == 0)
		{
			matMulGlobalLaunch(threads, blocks, width);
		}
		else if (kernelName.rfind("matMulShared", 0) == 0)
		{
			matMulSharedLaunch(threads, blocks, width);
		}
		else
		{
			std::cout << op << "\n";
		}
	}
}
