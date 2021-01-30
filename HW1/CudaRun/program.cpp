#include <iostream>

#include "CudaModule.h"
#include "MatAddOperation.h"
#include "DotProductOperation.h"
#include "DgemvOperation.h"

#include <device_launch_parameters.h>

void matAdd()
{
    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "matAdd.cu";
    CUfunction matAdd;

    cudaModule.Compile(kernelFile);

    MatAddOperation matAddOperation(1024, 1024);

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

    DotProductOperation<float> dotProductFloatOperation(1024);
    dotProduct = cudaModule.GetFunction("dotProductFloat");
    dotProductFloatOperation.Process(dotProduct);
    std::cout << "test passed: dotProduct float" << std::endl;

    DotProductOperation<float2> dotProductFloat2Operation(1234);
    dotProduct = cudaModule.GetFunction("dotProductFloat2");
    dotProductFloat2Operation.Process(dotProduct);
    std::cout << "test passed: dotProduct float2" << std::endl;

    DotProductOperation<float4> dotProductFloat4Operation(1234);
    dotProduct = cudaModule.GetFunction("dotProductFloat4");
    dotProductFloat4Operation.Process(dotProduct);
    std::cout << "test passed: dotProduct float4" << std::endl;
}

void dgemvOperation()
{
    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "dgemv.cu";
    CUfunction dgemv;

    cudaModule.Compile(kernelFile);

    DgemvOperation dgemvOperation(1024, 1024);
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
