#include <string>
#include <fstream>
#include <random>

#include "param_loader.h"

void load_params(std::string path, const int& arrays, param_array* paramArrays)
{
	std::ifstream istream(path, std::ios::binary | std::ios::in);

	if (!istream.is_open())
	{
		throw std::runtime_error("Cannot open file `" + path + "`!");
	}

	for (int i = 0; i < arrays; i++)
	{
		istream.read((char*)paramArrays[i].array, paramArrays[i].floats * sizeof(float));
	}

	istream.close();
}

void save_params(std::string path, const int& arrays, param_array* paramArrays)
{
	std::ofstream ostream(path, std::ios::binary | std::ios::out);

	if (!ostream.is_open())
	{
		throw std::runtime_error("Cannot open file `" + path + "`!");
	}


	for (int i = 0; i < arrays; i++)
	{
		ostream.write((char*)paramArrays[i].array, paramArrays[i].floats * sizeof(float));
	}

	ostream.close();
}

void init_params(const int& arrays, param_array* paramArrays, int* layer_sizes)
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
				paramArrays[array].array[j] = normal(generator);
			}
		}
		else
		{
			// bias - init to zeros
			memset(paramArrays[array].array, 0, layer_sizes[array] * sizeof(float));
		}
	}
}