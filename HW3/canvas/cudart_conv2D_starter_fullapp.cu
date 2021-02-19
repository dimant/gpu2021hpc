
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * CUDA Kernels
 *
 */
 // 2D Convolution kernel "naive" global memory
__global__ void conv_2D_allglobal(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx_x < num_col && idx_y < num_row)
    {
        float result = 0.f;
        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = idx_y + filter_row;
                int image_col = idx_x + filter_col;
                float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ? d_input[image_row * num_col + image_col] : 0.f;
                float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];
                result += image_value * filter_value;
            }
        }
        d_output[idx_y * num_col + idx_x] = result;
    }

}


using namespace std;


float gaussBlurFilter_7x7[49] = {
   0.0086f / 3.0f,    0.0198f / 3.0f,    0.0326f / 3.0f,    0.0386f / 3.0f,    0.0326f / 3.0f,    0.0198f / 3.0f,    0.0086f / 3.0f,
   0.0198f / 3.0f,    0.0456f / 3.0f,    0.0751f / 3.0f,    0.0887f / 3.0f,    0.0751f / 3.0f,    0.0456f / 3.0f,    0.0198f / 3.0f,
   0.0326f / 3.0f,    0.0751f / 3.0f,	  0.1239f / 3.0f,    0.1463f / 3.0f,    0.1239f / 3.0f,    0.0751f / 3.0f,    0.0326f / 3.0f,
   0.0386f / 3.0f,    0.0887f / 3.0f,    0.1463f / 3.0f,    0.1729f / 3.0f,    0.1463f / 3.0f,    0.0887f / 3.0f,    0.0386f / 3.0f,
   0.0326f / 3.0f,    0.0751f / 3.0f,    0.1239f / 3.0f,	   0.1463f / 3.0f,    0.1239f / 3.0f,    0.0751f / 3.0f,    0.0326f / 3.0f,
   0.0198f / 3.0f,    0.0456f / 3.0f,    0.0751f / 3.0f,    0.0887f / 3.0f,    0.0751f / 3.0f,    0.0456f / 3.0f,    0.0198f / 3.0f,
   0.0086f / 3.0f,    0.0198f / 3.0f,    0.0326f / 3.0f,    0.0386f / 3.0f,    0.0326f / 3.0f,    0.0198f / 3.0f,    0.0086f / 3.0f
};

float gaussBlurFilter_5x5[25] = {
    1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f
};

float compositeLaplacianFilter[9] = {
    -1.0f, -1.0f, -1.0f,
    -1.0f, 9.0f, -1.0f,
    -1.0f, -1.0f, -1.0f
};

float basicLaplacianFilterDiags[9] = {
    1.0f, 1.0f, 1.0f,
    1.0f, -8.0f, 1.0f,
    1.0f, 1.0f, 1.0f
};

float sobelEdgeX[9] = {
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f
};


int filterWidth = 7;


/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int imgRows;
    int imgCols;
    int imgChannels;

    string strOutpath = "c:\\users\\photonuser\\Downloads\\curt_2Dconv_kernels\\";

    string inFile = "Big-Gray-Lena_8bit.png";
    
    unsigned char* h_inimg_char = stbi_load( (strOutpath + inFile).c_str(), &imgCols, &imgRows, &imgChannels, 0);
    //unsigned short *data16 = stbi_load_16("C:/Users/colinrein/Pictures/boat1_16bit.png", &imgRows, &imgCols, &imgChannels, 0);
    // REMEMBER: stbi_image_free(imgdata);

    int numElements = imgCols * imgRows;
    size_t size = numElements * sizeof(float);
    printf("[Convolution Filter of [%d x %d] image with filter]\n", imgRows, imgCols);

    float* h_inimg = (float*)malloc(size);
    for (int i = 0; i < numElements; i++)
    {
        h_inimg[i] = (int)h_inimg_char[i]; // convert from char to float, costly...
    }

    float* h_outimg = (float*)malloc(size);
    unsigned char* h_outimg_char = (unsigned char*)malloc(imgRows * imgCols * imgChannels * sizeof(unsigned char));
 
    // Verify that allocations succeeded
    if (h_inimg == NULL || h_outimg == NULL )
    {
        fprintf(stderr, "Failed to allocate host images!\n");
        exit(EXIT_FAILURE);
    }

	// Allocate memory for device input image, filter, output image
	// and make sure it is initialized with the correct data

    // Launch the CUDA Kernel
    dim3 threadsPerBlock{ 16,16,1 };
    dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
    printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
 
    conv_2D_allglobal <<< blocksPerGrid, threadsPerBlock >>> (d_outimg, d_inimg, d_filt, imgRows, imgCols, filterWidth);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

	// retrieve result image from device
 

    // convert float to char (lossy...)
    for (int i = 0; i < numElements; i++)
    {
        h_outimg_char[i] = (unsigned char)(unsigned int)h_outimg[i]; // convert from float to 8-bit char, costly...
    }


    string strOutfile = "result_image.png";
    int stbErr = stbi_write_jpg((strOutpath + strOutfile).c_str(), imgCols, imgRows, imgChannels, h_outimg_char, 100);

    // Free device global memory


    // Free host memory
    stbi_image_free(h_inimg);
    free(h_outimg);

    printf("Done\n");
    return 0;
}
