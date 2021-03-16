#include "device_launch_parameters.h"
#include "nn_cuda_helper.cuh"

//===================================================================================================
//============== CUDA KERNELS ================
__global__ void nn_layer_fullyconnected_forward(float* X, float* W, float* b, float* Z, float* Y)
{
    //INPUTS
     // C 
     // K
     // w_tile_grid
     // X
     // W
     // b
    // OUTPUTS
    // Y : output FM
    // Z : pre-activation output FM
    int ns, m;

    // TODO: assign thread indices for thread-specific work based on CUDA grid execution configuration

    // CONCEPTUAL MODEL: Z = W.X + b  : matrix mult

    // CONCEPTUAL MODEL: Y = f(Z) : activation function
}

__global__ void nn_layer_fullyconnected_backward(float* dY, float* W, float* Aprev, float* Zprev, float* dX, float* dW, float* db)
{
    //INPUTS
    // dY
    // W
    // Aprev
    // Zprev
    // OUTPUTS
    // dX
    // dW
    // db

    // TODO: assign thread indices for thread-specific work based on CUDA grid execution configuration

    // CONCEPTUAL MODEL: db = dY

    // CONCEPTUAL MODEL: dW = dY.A'  : matrix mult

    // CONCEPTUAL MODEL: Y = f(Z) : activation function
}

__global__ void nn_layer_output_layer_backward(float* Targets, float* Ypred, float* Zout, float* dY3)
{
    //INPUTS
    // targets = labeled target values, known outputs for each input sample
    // Ypred = (network predictions) output results from final layer 
    // Zout = pre-activation value of output layer
    // OUTPUTS
    // dY3 = backprop value to pass to prior layer (layer 3)

    // CONCEPTUAL MODEL: dY3 = (Target - Ypred)* f(Zout)

}

__global__ void nn_layer_conv_forward(int C, int K, int w_tile_grid, float* X, float* W, float* Y, float* Z)
{
    //INPUTS
    // C = number of input
    // K = filter width (square filter W=H=K)
    // w_tile_grid
    // X
    // W = filter weights, 1D flattened array
    // OUTPUTS
    // Y : output FM
    // Z : pre-activation output FM
    int b, m, h, w, c, p, q;

    // Assign thread indices for thread-specific work based on CUDA grid execution configuration
    b = blockIdx.x;    // batch sample #
    m = blockIdx.y;    // output feature map #
    h = blockIdx.z / w_tile_grid + threadIdx.y;  // rows of 2D thread block
    w = blockIdx.z % w_tile_grid + threadIdx.x;  // cols of 2D thread block

    // CONCEPTUAL MODEL: Y = 
/*    float acc = 0;
    for (c = 0; c < C; ++c)
        for (p = 0; p < K; ++p)
            for (q = 0; q < K; ++q)
  */              // TODO: convert indexing into 1D linear
  //acc += X[b, c, h + p, w + q] * W[m, c, p, q];

    // TODO: convert indexing into 1D linear
    //Y[b, m, h, w] = acc;  // single output element per thread

}

__global__ void nn_layer_conv_backward()
{

}

__global__ void nn_layer_pooling_forward(float* X, float* W, float* Y)
{

}

__global__ void nn_layer_pooling_backward()
{

}

//=======================================================
// EXAMPLEs of how to use the Activation Functions
__global__ void nnTestKernelActivations(float* in, float* result)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int i = idx * 4;
    result[i] = sigmoid(in[i]);
    result[i + 1] = sigmoidPrime(in[i + 1]);
    result[i + 2] = tanh(in[i + 2]);
    result[i + 3] = tanhPrime(in[i + 3]);
}

__global__ void nnTestSoftmax10(float* in, float* result)
{
    int idx = threadIdx.x;

    int i = idx * 2;
    result[i] = softmax10(&in[0], idx);
    result[i + 1] = softmax10Prime(&in[0], idx, idx);
}