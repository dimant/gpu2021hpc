#include <assert.h>
#include <iostream>

#include "cudart_util.h"
#include "fc_forward.h"

#include "unit_tests.h"

void test_fc_forward_relu()
{
	int perceptive_field = 10;
	int output_field = 2;

	float* data;
	cudaCheckError(cudaMallocManaged(&data, perceptive_field * sizeof(float)));

	float* weights;
	cudaCheckError(cudaMallocManaged(&weights, perceptive_field * output_field * sizeof(float)));

	float* bias;
	cudaCheckError(cudaMallocManaged(&bias, perceptive_field * output_field * sizeof(float)));

	float* Z = nullptr;
	cudaCheckError(cudaMallocManaged(&Z, output_field * sizeof(float)));

	float* Y = nullptr;
	cudaCheckError(cudaMallocManaged(&Y, output_field * sizeof(float)));

	for (int i = 0; i < perceptive_field; i++)
	{
		data[i] = 1.0f;
	}

	for (int i = 0; i < output_field; i++)
	{
		for (int j = 0; j < perceptive_field; j++)
		{
			weights[i * perceptive_field + j] = 1.0f / (float)(i + 1.0f);
			bias[i * perceptive_field + j] = 100.0f;
		}
	}

	fc_forward CUDA_KERNEL(1, output_field) (data, weights, bias, Z, Y, perceptive_field, ACT_RELU);
	cudaDeviceSynchronize();

	assert(Z[0] - 110.0f < 0.00001f);
	assert(Z[1] - 105.0f < 0.00001f);
	assert(Y[0] - 110.0f < 0.00001f);
	assert(Y[1] - 105.0f < 0.00001f);
}

void test_fc_forward_softmax10()
{
	int perceptive_field = 10;
	int output_field = 2;

	float* data;
	cudaCheckError(cudaMallocManaged(&data, perceptive_field * sizeof(float)));

	float* weights;
	cudaCheckError(cudaMallocManaged(&weights, perceptive_field * output_field * sizeof(float)));

	float* bias;
	cudaCheckError(cudaMallocManaged(&bias, perceptive_field * output_field * sizeof(float)));

	float* Z = nullptr;
	cudaCheckError(cudaMallocManaged(&Z, output_field * sizeof(float)));

	float* Y = nullptr;
	cudaCheckError(cudaMallocManaged(&Y, output_field * sizeof(float)));

	for (int i = 0; i < perceptive_field; i++)
	{
		data[i] = 1.0f;
	}

	for (int i = 0; i < output_field; i++)
	{
		for (int j = 0; j < perceptive_field; j++)
		{
			weights[i * perceptive_field + j] = 1.0f / (float)(i + 1.0f);
			bias[i * perceptive_field + j] = 0.0f;
		}
	}

	fc_forward CUDA_KERNEL(1, output_field) (data, weights, bias, Z, Y, perceptive_field, ACT_SOFTMAX10);
	cudaDeviceSynchronize();

	assert(Z[0] - 110.0f < 0.00001f);
	assert(Z[1] - 105.0f < 0.00001f);

	assert(Y[0] - 0.992948890 < 0.00001f);
	assert(Y[1] - 0.00669043697 < 0.00001f);

	assert(Y[0] + Y[1] - 1.0f < 0.00001f);
}

void unit_tests()
{
	test_fc_forward_relu();
	test_fc_forward_softmax10();

	std::cout << "All tests passed." << std::endl;
}