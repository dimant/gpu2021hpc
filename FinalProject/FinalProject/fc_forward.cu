#include <math.h>
#include <stdio.h>

#include <device_launch_parameters.h>

#include "fc_forward.h"

__device__ __host__ __forceinline__ float relu(float f)
{
    return (fmaxf(0, f));
}

__device__ __host__ __forceinline__ float reluPrime(float f)
{
    if (0.f == f)
        return 0.5f;
    else
        return (f < 0.f ? 0.0f : 1.0f);
}

__device__ __host__ __forceinline__ float softmax10(float* fvec10, int ti)
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

// fvec10: activations from previous layer
// ds_i: perceptron index within the layer
// ds_k: index of vector element towards which we are taking a partial derivative
__device__ __host__ __forceinline__ float softmax10Prime(float* fvec10, int ds_i)
{
    float si = softmax10(fvec10, ds_i);
    return si * (1 - si);
}


__global__ void fc_forward(const float* X, const float* W, const float* b, float* Z, float* Y, const int n, const int activation)
{
    int wi = threadIdx.x;

    float sum = b[wi];

    for (int xi = 0; xi < n; xi++)
    {
        sum += X[xi] * W[wi * n + xi];
    }

    Z[wi] = sum;

    switch (activation)
    {
        case ACT_RELU:
            Y[wi] = relu(sum);
            break;
        case ACT_SOFTMAX10:
            __syncthreads();
            Y[wi] = softmax10(Z, wi);
            break;
    }
}

__global__ void output_backward(float* labels, float* Yhat, float* Z, int n, float* dY)
{
    int ds_i = threadIdx.x;

    dY[ds_i] = (labels[ds_i] - Yhat[ds_i]) * softmax10Prime(Z, ds_i);
}

// last layer: deltaL = (y hat - y) activation function'(previous layer activation)
// any other layer: 
// https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
// https://cloudacademy.com/course/data-machine-learning-gradient-descent/derivative-calculation/
// https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c