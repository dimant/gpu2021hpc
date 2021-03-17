#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#define ACT_RELU      0
#define ACT_TANH      1
#define ACT_SIGMOID   2
#define ACT_SOFTMAX   3
#define ACT_SOFTMAX10 4

__device__ __host__ __forceinline__ float relu(float f)
{
    return (fmaxf(0, f));
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

extern "C" __global__ void fc_forward(float* X, float* W, float* b, float* Z, float* Y, int n, int activation)
{
    int wi = threadIdx.x;

    Z[wi] = b[wi];

    for (int xi = 0; xi < n; xi++)
    {
        Z[wi] += X[xi] * W[wi * n + xi];
    }

    switch (activation)
    {
        case ACT_RELU:
            Y[wi] = relu(Z[wi]);
            break;
        case ACT_SOFTMAX10:
            Y[wi] = relu(Z[wi]);
            break;
    }

}
