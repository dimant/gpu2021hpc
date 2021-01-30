#include <iostream>

#include "CudaModule.h"
#include "MatAddOperation.h"
#include "DotProductOperation.h"

void matAdd()
{
    const char* kernelFile = "matAdd.cu";

    CudaModule cudaModule;
    cudaModule.Init();
    cudaModule.Compile(kernelFile);

    MatAddOperation matAddOperation(10, 10);

    CUfunction matAdd = cudaModule.GetFunction("matAdd");
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
    // dot product: https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
    // https://github.com/jiekebo/CUDA-By-Example/blob/master/5-dotproduct.cu
    // http://cuda-programming.blogspot.com/2013/01/vector-dot-product-in-cuda-c.html
    // https://stackoverflow.com/questions/28095261/float4-not-faster-than-float-in-cuda

    const char* kernelFile = "dotProduct.cu";

    CudaModule cudaModule;
    cudaModule.Init();
    cudaModule.Compile(kernelFile);

    DotProductOperation<float> dotProductOperation(1234);

    CUfunction dotProduct = cudaModule.GetFunction("dotProductFloat");
    dotProductOperation.Process(dotProduct);
    std::cout << "test passed: dotProductFloat" << std::endl;
}

int main(int argc, char** argv)
{
    dotProduct();

    return 0;
}
