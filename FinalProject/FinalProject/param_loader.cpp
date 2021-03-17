#include <string>
#include <fstream>
#include <random>

#include "param_loader.h"

void load_params(std::string path, const size_t& arrays, param_array* param_arrays)
{
	std::ifstream istream(path, std::ios::binary | std::ios::in);

	if (!istream.is_open())
	{
		throw std::runtime_error("Cannot open file `" + path + "`!");
	}

	for (int i = 0; i < arrays; i++)
	{
		istream.read((char*)param_arrays[i].array, param_arrays[i].floats * sizeof(float));
	}

	istream.close();
}

void save_params(std::string path, const size_t& arrays, param_array* param_arrays)
{
	std::ofstream ostream(path, std::ios::binary | std::ios::out);

	if (!ostream.is_open())
	{
		throw std::runtime_error("Cannot open file `" + path + "`!");
	}


	for (int i = 0; i < arrays; i++)
	{
		ostream.write((char*)param_arrays[i].array, param_arrays[i].floats * sizeof(float));
	}

	ostream.close();
}

void init_param_array(param_array* param_array, int floats)
{
	param_array->array = new float[floats];
	param_array->grad_array = new float[floats];
}

void destroy_param_array(param_array* pa)
{
	delete[] pa->array;
	delete[] pa->grad_array;
}

void init_params(const size_t& arrays, param_array* param_arrays, int* layer_sizes)
{
	std::default_random_engine generator;

	for (int array = 0; array < arrays; array++)
	{
		float stddev = 0;

		if (0 == array)
		{
			stddev = sqrtf(2.0f / 784); // He initializer
		}
		else
		{
			stddev = sqrt(2.0f / layer_sizes[array - 1]);
		}

		if (0 == (array % 2)) // weight arrays are even indices
		{
			std::normal_distribution<float> normal(0.0f, stddev);

			for (int j = 0; j < layer_sizes[array]; j++)
			{
				param_arrays[array].array[j] = normal(generator);
			}
		}
		else
		{
			// bias - init to zeros
			memset(param_arrays[array].array, 0, layer_sizes[array] * sizeof(float));
		}
	}
}