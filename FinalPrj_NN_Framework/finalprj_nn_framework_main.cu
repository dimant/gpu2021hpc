
#include "cuda_runtime.h"
#include "nn_cuda_kernels.cuh"

#include <stdio.h>
#include <cassert>
#include <random>

inline cudaError_t CURT_CHK(cudaError_t curtRes)
{
    if (cudaSuccess != curtRes)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(curtRes));
        assert(curtRes == cudaSuccess);
    }
    return curtRes;
}

#include "mnist_dataset_helper.h"
#include "nn_host_helper.h"

#define LOAD_NETWORK_PARAMETERS  0


// forward-declarations
void run_fullyconnected_neuralnet(bool bLoadParams, bool bSaveParams, float* mnist_input, float* mnist_targets, int num_samples);
void run_convolutional_neuralnet(bool bLoadParams, bool bSaveParams, float* mnist_input, float* mnist_targets, int num_samples);

//===================================================================================================
//============== Main Host Application ================
int main()
{
    // =========== LOAD MNIST datasets ================================
    // Assume MNIST dataset is in C:\FinalProject\mnist sub-folders
    // TODO: update full-paths if necessary
    string mnist_test_imgs_filepath("C:\\FinalProject\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte");
    string mnist_test_labels_filepath("C:\\FinalProject\\mnist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte");
    string mnist_train_imgs_filepath("C:\\FinalProject\\mnist\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
    string mnist_train_labels_filepath("C:\\FinalProject\\mnist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");

    // Functions will malloc memory for arrays, host/caller must free() it.
    // these int values will be filled with what is read from the MNIST dataset files
    int num_test_imgs = 0;
    int num_train_imgs = 0;
    int num_test_lbls = 0;
    int num_train_lbls = 0;
    int n_rows = 0;
    int n_cols = 0;
    //// loads data into one contiguous 1D packed memory region, sample-by-sample, row-major layout
    // each sample label data is a one-hot-encoded vector of length = 10 where the "correct" digit = 1 and all others = 0
    float* X0_test = NULL;  // test images = input
    float* T_test = NULL;  // test labels (output targets)
    float* X0_train = NULL; // train images = input
    float* T_train = NULL; // train labels (output targets)

    load_preproc_mnist_labels(mnist_test_labels_filepath, &T_test, num_test_lbls); // test set - targets
    load_preproc_mnist_labels(mnist_train_labels_filepath, &T_train, num_train_lbls); // train set - targets
    load_and_preproc_mnist_images(mnist_test_imgs_filepath, &X0_test, num_test_imgs, n_rows, n_cols); // test set - images
    load_and_preproc_mnist_images(mnist_train_imgs_filepath, &X0_train, num_train_imgs, n_rows, n_cols); // train set - images
    //===============================================
 
    // 
    // TODO: Choose which GPU to run on, make sure this agrees with your system (works for AWS VDE).
    CURT_CHK(cudaSetDevice(0));

    //===== TRAINING
    run_fullyconnected_neuralnet(false,true, X0_train, T_train, num_train_imgs);
    //run_convolutional_neuralnet(false, false, X0_train, T_train);

    //===== TESTING (after training is done and final network parameters are saved)
    //run_fullyconnected_neuralnet(false, false, X0_test, T_test);
    //run_convolutional_neuralnet(false, false, X0_test, T_test);

    //===============================================

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CURT_CHK(cudaDeviceReset());

    return 0;

};


void run_fullyconnected_neuralnet(bool bLoadParams, bool bSaveParams, float* mnist_input, float* mnist_targets, int num_samples)
{
    // A simple fully-connected NN architecture
    // INPUT layer (0) : 28x28 MNIST image
    // FC layer 1 : 128 nodes, sigmoid or relu activation :             100,352 weights, 128 bias
    // FC layer 2 : 128 nodes, sigmoid or relu  activation               16,512 weights, 128 bias
    // Output FC layer (3) : 10 nodes, softmax activation                  1280 weights, 10 bias
    // Expected test accuracy ~ 99%

    const int NUM_NN_PARAM_ARRAYS = 6;

    string strNNparamfile("C:\\FinalProject\\my_FC_NN_params.bin");  // USE a unique name which you can remember which network arch it's for!

    // Define layer sizes array with the size (in # floats) for each parameter: weights and biases get separate entries.
    int layer_sizes[] = { 100352, 128, 16384, 128, 1280, 10 };  // W1, b1, W2, b2, W3, b3
    
    // allocate memory for the array of layerParamArray structures - which define the size and base address of each weight/bias array
    layerParamArray* pParamArrayLoad = (layerParamArray*)malloc(NUM_NN_PARAM_ARRAYS * sizeof(layerParamArray));
    for (int i = 0; i < NUM_NN_PARAM_ARRAYS; ++i)
    {
        pParamArrayLoad[i].num_floats = layer_sizes[i];
        // NOTE: these arrays get passed to GPU so allocate with CUDA memory (unified managed memory)
        CURT_CHK(cudaMallocManaged(&(pParamArrayLoad[i].pArray), layer_sizes[i] * sizeof(float)));
        // also allocate array of same size for error gradient used during backprop (will not be persisted with network parameters)
        CURT_CHK(cudaMallocManaged(&(pParamArrayLoad[i].pGradArray), layer_sizes[i] * sizeof(float)));
    }

    if (bLoadParams) // LOAD existing save network parameters
    {
        //=== LOADING network parameters from a binary file

        if (0 != load_network_parameters_from_binaryfile(strNNparamfile, NUM_NN_PARAM_ARRAYS, pParamArrayLoad))
        {
            fprintf(stderr, "Error loading network parameters from file: %s.\n", strNNparamfile.c_str());
        }
    }
    else  // INIT new params
    {
       default_random_engine generator;
       for (int i = 0; i < NUM_NN_PARAM_ARRAYS; ++i)
        {
           float stddev = 0;
           if (0 == i)
               stddev = sqrtf(2.0f / 784); // He initializer
           else
               stddev = sqrt(2.0f / layer_sizes[i - 1]);

           if (0 == (i % 2)) // weight arrays are even indices
           {
               normal_distribution<float> normD(0.0f, stddev);
               for (int j = 0; j < layer_sizes[i]; ++j)
               {
                   pParamArrayLoad[i].pArray[j] = normD(generator);
               }
           }
           else // bias - init to zeros
               CURT_CHK(cudaMemset(pParamArrayLoad[i].pArray, 0, layer_sizes[i] * sizeof(float)));
        }

    }

    // Set up arrays for intermediate network data values (used for backprop)
    float* Z1 = NULL, float* Z2 = NULL, float* Z3 = NULL;
    float* Y1 = NULL, float* Y2 = NULL, float* Y3 = NULL;
    float* dOut = NULL, float* dY3 = NULL, float* dY2 = NULL, float* dY1 = NULL;

    CURT_CHK(cudaMallocManaged(&Z1, 128 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&Z2, 128 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&Z3, 10 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&Y1, 128 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&Y2, 128 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&Y3, 10 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&dY3, 10 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&dY2, 128 * sizeof(float)));
    CURT_CHK(cudaMallocManaged(&dY1, 128 * sizeof(float)));

    // TODO: MAIN FC NN Implementation here
    for(int s = 0; s < num_samples; ++s)
    {
        // Forward pass
        // TODO: define correct kernel execution configs for each layer, based on layer cuda kernel implementations
        // TODO: add additional input parameters to kernels if necessary
        dim3 numBlocks;
        dim3 numThreadsPerBlock;
        nn_layer_fullyconnected_forward << <numBlocks, numThreadsPerBlock >> > (mnist_input, pParamArrayLoad[0].pArray, pParamArrayLoad[1].pArray, Z1, Y1);
        nn_layer_fullyconnected_forward << <numBlocks, numThreadsPerBlock >> > (Y1, pParamArrayLoad[2].pArray, pParamArrayLoad[3].pArray, Z2, Y2);
        nn_layer_fullyconnected_forward << <numBlocks, numThreadsPerBlock >> > (Y2, pParamArrayLoad[4].pArray, pParamArrayLoad[5].pArray, Z3, Y3);

        //// Backward pass
        // TODO: define correct kernel execution configs for each layer, based on layer cuda kernel implementations
        // TODO: add additional input parameters to kernels if necessary
        nn_layer_output_layer_backward << <numBlocks, numThreadsPerBlock >> > (mnist_targets, Y3, Z3, dY3);
        nn_layer_fullyconnected_backward << <numBlocks, numThreadsPerBlock >> > (dY3, pParamArrayLoad[4].pArray, Y2, Z2, dY2, pParamArrayLoad[4].pGradArray, pParamArrayLoad[5].pGradArray  );
        nn_layer_fullyconnected_backward << <numBlocks, numThreadsPerBlock >> > (dY2, pParamArrayLoad[2].pArray, Y1, Z1, dY1, pParamArrayLoad[2].pGradArray, pParamArrayLoad[3].pGradArray);
        nn_layer_fullyconnected_backward << <numBlocks, numThreadsPerBlock >> > (dY1, pParamArrayLoad[0].pArray, mnist_input, NULL, NULL, pParamArrayLoad[0].pGradArray, pParamArrayLoad[1].pGradArray);

        // TODO: put numerical gradient checking here - be sure to disable for real complete training & test runs

    }

    if (bSaveParams)
    {
        //=== SAVING all network parameters to a binary file
         
        if (0 != save_network_parameters_to_binaryfile(strNNparamfile, NUM_NN_PARAM_ARRAYS, pParamArrayLoad))
        {
            fprintf(stderr, "Error saving network parameters to file: %s.\n", strNNparamfile.c_str());
        }
    }

};

void run_convolutional_neuralnet(bool bLoadParams, bool bSaveParams, float* mnist_input, float* mnist_targets, int num_samples)
{
    // A simple convolutional NN architecture
    // INPUT layer (0): 28x28 MNIST image
    // CONV layer 1 : 32 output FM, 3x3 filter, relu activation:        288 weights, 32 bias
    // MAX POOL layer 2 : 2x2 pooling                                   0
    // CONV layer 3 : 64 output FM, 3x3 filter, relu activation         18432 weights, 64 bias
    // MAX POOL layer 4 : 2x2 pooling                                   0
     // OUTPUT FC layer (5) : 10 nodes, softmax activation:             16000 weights, 10 bias  
    // Expected test accuracy ~ %99

    float* W1 = NULL, float* W2 = NULL, float* W3 = NULL, float* b1 = NULL, float* b2 = NULL, float* b3 = NULL;

    string strNNparamfile("C:\\FinalProject\\my_CONV_NN_params.bin");  // USE a unique name which you can remember which network arch it's for!
    const int NUM_NN_PARAM_ARRAYS = 6;
    // Set up layer sizes array with the size (in # floats) for each parameter: weights and biases get separate entries.
    int layer_sizes[] = { 288, 32, 18432, 64, 16000, 10 };

    // allocate memory for the array of layerParamArray structures - which define the size and base address of each weight/bias array
    layerParamArray* pParamArrayLoad = (layerParamArray*)malloc(NUM_NN_PARAM_ARRAYS * sizeof(layerParamArray));
    for (int i = 0; i < NUM_NN_PARAM_ARRAYS; ++i)
    {
        pParamArrayLoad[i].num_floats = layer_sizes[i];
        // NOTE: these arrays get passed to GPU so allocate with CUDA memory (unified managed memory)
        CURT_CHK(cudaMallocManaged(&(pParamArrayLoad[i].pArray), layer_sizes[i] * sizeof(float)));
    }

    if (bLoadParams)
    {
        //=== LOADING network parameters from a binary file
        if (0 != load_network_parameters_from_binaryfile(strNNparamfile, NUM_NN_PARAM_ARRAYS, pParamArrayLoad))
        {
            fprintf(stderr, "Error loading network parameters from file: %s.\n", strNNparamfile.c_str());
        }
    }
    else  // INIT new params
    {

    }

    // TODO: MAIN CONV NN Implementation here

    if (bSaveParams)
    {
        //=== SAVING network parameters to a binary file
        // pass the pointer to base address of each parameter array (pParams is an array of pointers)
        if (0 != save_network_parameters_to_binaryfile(strNNparamfile, NUM_NN_PARAM_ARRAYS, pParamArrayLoad))
        {
            fprintf(stderr, "Error saving network parameters to file: %s.\n", strNNparamfile.c_str());
        }
    }

};

