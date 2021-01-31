#include <iostream>

#include "CudaModule.h"
#include "OpenCLModule.h"

#include "MatAddCuda.h"
#include "MatAddOpenCL.h"

#include "DotProductCuda.h"

#include "DgemvCuda.h"

void matAddCuda()
{
    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "matAdd.cu";

    cudaModule.Compile(kernelFile);

    MatAddCuda matAddOperation(4096, 4096);
    CudaContext context = cudaModule.GetContext("matAdd");
    matAddOperation.Process(context);
    std::cout << "Cuda test passed: matAdd" << std::endl;

    context = cudaModule.GetContext("matAddRow");
    matAddOperation.Process(context);
    std::cout << "Cuda test passed: matAddRow" << std::endl;

    context = cudaModule.GetContext("matAddCol");
    matAddOperation.Process(context);
    std::cout << "Cuda test passed: matAddCol" << std::endl;
}

void matAddOpenCL()
{
    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "matAdd.cl";

    openclModule.Compile(kernelFile);

    MatAddOpenCL matAddOperation(10, 10);
    OpenCLContext context = openclModule.GetContext("matAdd");
    matAddOperation.Process(context);
    std::cout << "OpenCL test passed: matAdd" << std::endl;
}

void dotProductCuda()
{
    CudaModule cudaModule;
    cudaModule.Init();

    // dot product: https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
    // https://github.com/jiekebo/CUDA-By-Example/blob/master/5-dotproduct.cu
    // http://cuda-programming.blogspot.com/2013/01/vector-dot-product-in-cuda-c.html
    // https://stackoverflow.com/questions/28095261/float4-not-faster-than-float-in-cuda

    const char* kernelFile = "dotProduct.cu";
    CUfunction dotProduct;

    cudaModule.Compile(kernelFile);

    DotProductCuda<float> dotProductFloat(16777216);
    CudaContext context = cudaModule.GetContext("dotProductFloat");
    dotProductFloat.Process(context);
    std::cout << "Cuda test passed: dotProduct float" << std::endl;

    DotProductCuda<float2> dotProductFloat2(16777216);
    context = cudaModule.GetContext("dotProductFloat2");
    dotProductFloat2.Process(context);
    std::cout << "Cuda test passed: dotProduct float2" << std::endl;

    DotProductCuda<float4> dotProductFloat4(16777216);
    context = cudaModule.GetContext("dotProductFloat4");
    dotProductFloat4.Process(context);
    std::cout << "Cuda test passed: dotProduct float4" << std::endl;
}

void dgemvCuda()
{
    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "dgemv.cu";
    CUfunction dgemv;

    cudaModule.Compile(kernelFile);

    DgemvCuda dgemvOperation(4096, 4096*2);
    CudaContext context = cudaModule.GetContext("dgemv");
    dgemvOperation.Process(context);
    std::cout << "Cuda test passed: dgemv" << std::endl;
}

int main(int argc, char** argv)
{
    //matAddCuda();
    //dotProductCuda();
    //dgemvCuda();

    matAddOpenCL();

    return 0;
}
