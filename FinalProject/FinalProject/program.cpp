#include <vector>
#include <string>

#include "popl.h"

#include "mnist_loader.h"
#include "param_loader.h"


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

void test_net(std::string data_dir, std::string weights_fname)
{
	float* images = load_images(data_dir + mnist_test_images_file);
	float* labels = load_labels(data_dir + mnist_test_labels_file);

	const size_t arrays = (size_t)layers * 2;
	param_array* param_arrays = create_param_arrays(arrays);
	load_params(data_dir + weights_fname, arrays, param_arrays);
}

void save_init_params(std::string data_dir, std::string fname)
{
	const size_t arrays = (size_t)layers * 2;
	param_array* param_arrays = create_param_arrays(arrays);

	init_params(arrays, param_arrays, layer_sizes);

	save_params(data_dir + fname, arrays, param_arrays);
}

int main(int argc, char** argv)
{

	popl::OptionParser op("Final Project options");

	auto help_option = op.add<popl::Switch>("h", "help", "produce help message");
	auto data_dir_option = op.add<popl::Value<std::string>>("d", "data-dir", "Directory containing extracted MNIST data.");
	auto save_init_option = op.add<popl::Value<std::string>>("i", "save-init", "Init weights, save to given file name.");
	auto test_option = op.add<popl::Value<std::string>>("t", "test-net", "Test network using given weights");

	op.parse(argc, argv);

	if(test_option->is_set() && data_dir_option->is_set())
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
