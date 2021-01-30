#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <device_launch_parameters.h>

#include <cuda.h>
#include "cuda_util.h"

void allocateHostVectors(float* &A, float* &B, float* &C, size_t nrows, size_t ncols)
{
    size_t size = nrows * ncols * sizeof(float);

    A = (float*) malloc(size);
    B = (float*) malloc(size);
    C = (float*) malloc(size);

    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "failed to allocate host vectors");
        exit(EXIT_FAILURE);
    }
}

void allocateDeviceVectors(CUdeviceptr* A, CUdeviceptr* B, CUdeviceptr* C, size_t nrows, size_t ncols)
{
    size_t size = nrows * ncols * sizeof(float);

    checkCudaError(cuMemAlloc(A, size));
    checkCudaError(cuMemAlloc(B, size));
    checkCudaError(cuMemAlloc(C, size));
}

void initCuda(const char* kernelFile, const char* kernelName, CUfunction *cuFunction, CUcontext* cuContext)
{
    int deviceCount = 0;

    checkCudaError(cuInit(0));

    checkCudaError(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    CUdevice cuDevice;
    CUmodule cuModule;

    checkCudaError(cuDeviceGet(&cuDevice, 0));

    checkCudaError(cuCtxCreate(cuContext, 0, cuDevice));

    setCwdToExeDir();

    char* cubinResult;
    size_t cubinResultSize;

    cudaCompileKernel(cuDevice, kernelFile, &cubinResult, &cubinResultSize);

    checkCudaError(cuModuleLoadData(&cuModule, cubinResult));

    delete[] cubinResult;

    checkCudaError(cuModuleGetFunction(cuFunction, cuModule, kernelName));
}

int main(int argc, char** argv)
{
    const char* kernelFile = "matAdd.cu";
    const char* kernelName = "matAddCol";

    CUcontext cuContext;
    CUfunction matAdd;

    initCuda(kernelFile, kernelName, &matAdd, &cuContext);


    float* h_A;
    float* h_B;
    float* h_C;

    CUdeviceptr d_A;
    CUdeviceptr d_B;
    CUdeviceptr d_C;

    size_t nrows = 10;
    size_t ncols = 10;
    size_t size = nrows * ncols * sizeof(float);

    allocateHostVectors(h_A, h_B, h_C, nrows, ncols);
    allocateDeviceVectors(&d_A, &d_B, &d_C, nrows, ncols);

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            h_A[i * ncols + j] = i;
            h_B[i * ncols + j] = j;
        }
    }

    checkCudaError(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaError(cuMemcpyHtoD(d_B, h_B, size));

    unsigned int threads = 1;
    dim3 blockSize(threads, threads);
    dim3 gridSize(1, 1);

    void* args[5] = { &d_A, &d_B, &d_C, &nrows, &ncols};
    checkCudaError(cuLaunchKernel(matAdd,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        threads * threads * sizeof(float),
        NULL, args, NULL));

    checkCudaError(cuMemcpyDtoH(reinterpret_cast<void*>(h_C), d_C, size));

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            double a = h_A[i * ncols + j];
            double b = h_B[i * ncols + j];
            double c = h_C[i * ncols + j];
            if (fabs(a + b - c) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d,%d!\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }

    std::cout << "Test Passed." << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cuMemFree(d_A));
    checkCudaError(cuMemFree(d_B));
    checkCudaError(cuMemFree(d_C));
    checkCudaError(cuCtxDestroy(cuContext));

    return 0;
}
