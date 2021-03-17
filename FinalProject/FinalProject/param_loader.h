#pragma once

#include <string>

struct param_array
{
	int floats;
	float* array;
	float* grad_array;
};

void load_params(std::string path, const size_t& arrays, param_array* param_arrays);

void save_params(std::string path, const size_t& arrays, param_array* param_arrays);

void init_params(const size_t& arrays, param_array* param_arrays, int* layer_sizes);

void init_param_array(param_array* param_array, int floats);

void destroy_param_array(param_array* param_array);