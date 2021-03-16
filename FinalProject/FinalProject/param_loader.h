#pragma once

#include <string>

struct param_array
{
	int floats;
	float* array;
	float* grad_array;
};

void load_params(std::string path, const int& arrays, param_array* paramArrays);

void save_params(std::string path, const int& arrays, param_array* paramArrays);

void init_params(const int& arrays, param_array* paramArrays, int* layer_sizes);

