#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "fc_forward.h"

__device__ __host__ __forceinline__ float relu(const float f)
{
    return (fmaxf(0, f));
}

__device__ __host__ __forceinline__ float relu_prime(const float f)
{
    if (0.f == f)
        return 0.5f;
    else
        return (f < 0.f ? 0.0f : 1.0f);
}

__device__ __host__ __forceinline__ float softmax10(const float* fvec10, const int ti)
{
    float sum = 0;

    float target = expf(fvec10[ti]);

#pragma unroll
    for (int i = 0; i < 10; i++)
    {
        sum += expf(fvec10[i]);
    }
    return (target / sum);
}

__device__ __host__ __forceinline__ float softmax10_prime(const float* fvec10, const int ds_i)
{
    float si = softmax10(fvec10, ds_i);
    return si * (1 - si);
}

__global__ void fc_forward(const float* X, const float* W, const float* b, float* Z, float* Y, const int inputs, const int activation)
{
    int neuron = threadIdx.x;

    float sum = b[neuron];

    for (int input = 0; input < inputs; input++)
    {
        sum += X[input] * W[neuron * inputs + input];
    }

    Z[neuron] = sum;

    switch (activation)
    {
        case ACT_RELU:
            Y[neuron] = relu(sum);
            break;
        case ACT_SOFTMAX10:
            __syncthreads();
            Y[neuron] = softmax10(Z, neuron);
            break;
    }
}

// last layer: deltaL = (y hat - y) activation function'(previous layer activation)
__global__ void output_backward(
    const float* labels,
    const float* Yhat,
    const float* Z,
    float* dY)
{
    int neuron = threadIdx.x;

    dY[neuron] = (labels[neuron] - Yhat[neuron]) * softmax10_prime(Z, neuron);
}

// any other layer: 
// https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
// https://cloudacademy.com/course/data-machine-learning-gradient-descent/derivative-calculation/
// https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
__global__ void fc_backward(
    const float* dY, // errors for current layer
    const float* Yprev,
    const float* Zprev,
    const float* W,
    float* dW,
    float* dB,
    float* dX, // errors for next layer
    const int inputs,
    const int neurons)
{
    int neuron = threadIdx.x;

    for (int input = 0; input < inputs; input++)
    {
        dW[neuron * inputs + input] = dY[neuron] * Yprev[input];
        dX[input] += W[neuron * inputs + input] * dY[input];
    }

    dB[neuron] = dY[neuron];

    __syncthreads();

    if (neuron == 0)
    {
        for (int input = 0; input < inputs; input++)
        {
            dX[input] *= relu_prime(Zprev[input]);
        }
    }
}
