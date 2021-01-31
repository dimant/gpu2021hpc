#include <iostream>

#include "CudaModule.h"
#include "MatAddCuda.h"
#include "DotProductCuda.h"
#include "DgemvOperation.h"

#include <device_launch_parameters.h>

void matAdd()
{
    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "matAdd.cu";
    CUfunction matAdd;

    cudaModule.Compile(kernelFile);

    MatAddCuda matAddOperation(4096, 4096);

    matAdd = cudaModule.GetFunction("matAdd");
    matAddOperation.Process(matAdd);
    std::cout << "test passed: matAdd" << std::endl;

    matAdd = cudaModule.GetFunction("matAddRow");
    matAddOperation.Process(matAdd);
    std::cout << "test passed: matAddRow" << std::endl;

    matAdd = cudaModule.GetFunction("matAddCol");
    matAddOperation.Process(matAdd);
    std::cout << "test passed: matAddCol" << std::endl;
}

void dotProduct()
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
    dotProduct = cudaModule.GetFunction("dotProductFloat");
    dotProductFloat.Process(dotProduct);
    std::cout << "test passed: dotProduct float" << std::endl;

    DotProductCuda<float2> dotProductFloat2(16777216);
    dotProduct = cudaModule.GetFunction("dotProductFloat2");
    dotProductFloat2.Process(dotProduct);
    std::cout << "test passed: dotProduct float2" << std::endl;

    DotProductCuda<float4> dotProductFloat4(16777216);
    dotProduct = cudaModule.GetFunction("dotProductFloat4");
    dotProductFloat4.Process(dotProduct);
    std::cout << "test passed: dotProduct float4" << std::endl;
}

void dgemvOperation()
{
    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "dgemv.cu";
    CUfunction dgemv;

    cudaModule.Compile(kernelFile);

    DgemvOperation dgemvOperation(4096, 4096*2);
    dgemv = cudaModule.GetFunction("dgemv");
    dgemvOperation.Process(dgemv);
    std::cout << "test passed: dgemv" << std::endl;
}

int main(int argc, char** argv)
{
    matAdd();
    dotProduct();
    dgemvOperation();

    return 0;
}
