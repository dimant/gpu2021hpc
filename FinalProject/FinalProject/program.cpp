#include <vector>
#include <string>
#include <math.h>

#include <cuda_runtime.h>

#include "popl.h"

#include "mnist_loader.h"
#include "param_loader.h"
#include "cudart_util.h"

#include "fc_forward.h"
#include "unit_tests.h"

std::string mnist_test_images_file = "t10k-images.idx3-ubyte";
std::string mnist_test_labels_file = "t10k-labels.idx1-ubyte";
std::string mnist_train_images_file = "train-images.idx3-ubyte";
std::string mnist_train_labels_file = "train-labels.idx1-ubyte";

int layers = 3;
int layer_sizes[] = { 100352, 128, 16384, 128, 1280, 10 };  // W1, b1, W2, b2, W3, b3

param_array* create_param_arrays(size_t arrays)
{

	param_array* param_arrays = new param_array[arrays];

	for (int i = 0; i < arrays; i++)
	{
		param_arrays[i].floats = layer_sizes[i];
		init_param_array(&(param_arrays[i]), layer_sizes[i]);
	}

	return param_arrays;
}

void save_init_params(std::string data_dir, std::string fname)
{
	const size_t arrays = (size_t)layers * 2;
	param_array* param_arrays = create_param_arrays(arrays);

	init_params(arrays, param_arrays, layer_sizes);

	save_params(data_dir + fname, arrays, param_arrays);
}

float cross_entropy(float* a, float* b, int classes)
{
	float sum = 0.0f;

	for (int i = 0; i < classes; i++)
	{
		sum += a[i] * logf(b[i]);
	}

	return -sum;
}

void test_net(std::string data_dir, std::string weights_fname)
{
	data_dim dd;
	size_t n_labels;
	int classes = 10;

	float* _images = load_images(data_dir + mnist_test_images_file, dd);
	float* images;
	float* labels = load_labels(data_dir + mnist_test_labels_file, n_labels);

	cudaCheckError(cudaMallocManaged(&images, (size_t)dd.n * (size_t)dd.rows * (size_t)dd.cols * sizeof(float)));
	memcpy(images, _images, (size_t)dd.n * (size_t)dd.rows * (size_t)dd.cols * sizeof(float));

	const size_t arrays = (size_t)layers * 2;
	param_array* param_arrays = create_param_arrays(arrays);
	load_params(data_dir + weights_fname, arrays, param_arrays);

#define ALLOC_LAYER(idx) \
	float* Z ## idx; \
	float* Y ## idx; \
	cudaCheckError(cudaMallocManaged(&Z ## idx, layer_sizes[idx * 2 - 1] * sizeof(float))); \
	cudaCheckError(cudaMallocManaged(&Y ## idx, layer_sizes[idx * 2 - 1] * sizeof(float)));

	ALLOC_LAYER(1);
	ALLOC_LAYER(2);
	ALLOC_LAYER(3);

#undef ALLOC_LAYER

	int step = dd.rows * dd.cols;
	
	float loss = 0.0f;

	for (int i = 0; i < dd.n; i++)
	{
		fc_forward CUDA_KERNEL(1, 128) (images, param_arrays[0].array, param_arrays[1].array, Z1, Y1, step, ACT_RELU);
		cudaDeviceSynchronize();
		cudaCheckError(cudaGetLastError());

		fc_forward CUDA_KERNEL(1, 128) (Y1, param_arrays[2].array, param_arrays[3].array, Z2, Y2, step, ACT_RELU);
		cudaDeviceSynchronize();
		cudaCheckError(cudaGetLastError());

		fc_forward CUDA_KERNEL(1, 10) (Y2, param_arrays[4].array, param_arrays[5].array, Z3, Y3, step, ACT_SOFTMAX10);
		cudaDeviceSynchronize();
		cudaCheckError(cudaGetLastError());

		loss += cross_entropy(labels, Y3, classes) / dd.n;
	}

	std::cout << "Loss: " << loss << std::endl;
}

int main(int argc, char** argv)
{
	popl::OptionParser op("Final Project options");

	auto help_option = op.add<popl::Switch>("h", "help", "produce help message");
	auto unit_test_option = op.add<popl::Switch>("u", "unit-test", "run unit tests");
	auto data_dir_option = op.add<popl::Value<std::string>>("d", "data-dir", "Directory containing extracted MNIST data.");
	auto save_init_option = op.add<popl::Value<std::string>>("i", "save-init", "Init weights, save to given file name. (requires -d)");
	auto test_option = op.add<popl::Value<std::string>>("t", "test-net", "Test network using given weights (requires -d)");

	op.parse(argc, argv);

	if (unit_test_option->is_set())
	{
		unit_tests();
	}
	else if(test_option->is_set() && data_dir_option->is_set())
	{
		std::string data_dir = data_dir_option->value();

		test_net(data_dir, test_option->value());
	}
	else if (save_init_option->is_set() && data_dir_option->is_set())
	{
		save_init_params(data_dir_option->value(), save_init_option->value());
	}
	else
	{
		std::cout << op << "\n";
	}

	return 0;
}
