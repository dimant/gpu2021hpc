#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define ACT_RELU      0
#define ACT_TANH      1
#define ACT_SIGMOID   2
#define ACT_SOFTMAX   3
#define ACT_SOFTMAX10 4

__global__ void fc_forward(const float* X, const float* W, const float* b, float* Z, float* Y, const int n, const int activation);

