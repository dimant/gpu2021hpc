#include <iostream>

#include "CudaModule.h"
#include "OpenCLModule.h"

#include "MatAddCuda.h"
#include "MatAddOpenCL.h"

#include "DotProductCuda.h"
#include "DotProductOpenCL.h"

#include "DgemvCuda.h"
#include "DgemvOpenCL.h"

void matAddCuda()
{
    std::cout << std::endl << "[CUDA] MatAdd tests" << std::endl;

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
    std::cout << std::endl << "[OpenCL] MatAdd tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "matAdd.cl";

    openclModule.Compile(kernelFile);

    MatAddOpenCL matAddOperation(4096, 4096);
    OpenCLContext context = openclModule.GetContext("matAdd");
    matAddOperation.Process(context);
    std::cout << "OpenCL test passed: matAdd" << std::endl;

    context = openclModule.GetContext("matAddRow");
    matAddOperation.Process(context);
    std::cout << "OpenCL test passed: matAddRow" << std::endl;

    context = openclModule.GetContext("matAddCol");
    matAddOperation.Process(context);
    std::cout << "OpenCL test passed: matAddCol" << std::endl;
}

void dotProductCuda()
{
    std::cout << std::endl << "[Cuda] DotProduct tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    // dot product: https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
    // https://github.com/jiekebo/CUDA-By-Example/blob/master/5-dotproduct.cu
    // http://cuda-programming.blogspot.com/2013/01/vector-dot-product-in-cuda-c.html
    // https://stackoverflow.com/questions/28095261/float4-not-faster-than-float-in-cuda

    const char* kernelFile = "dotProduct.cu";

    cudaModule.Compile(kernelFile);
    CudaContext contextFloat = cudaModule.GetContext("dotProductFloat");
    CudaContext contextFloat2 = cudaModule.GetContext("dotProductFloat2");
    CudaContext contextFloat4 = cudaModule.GetContext("dotProductFloat4");

    std::cout << "large vector dotProduct float (A = 1.0 and B = 1.0)" << std::endl;
    DotProductCuda<float> dotProductFloatLarge(16777216);
    dotProductFloatLarge.Process(contextFloat);
    std::cout << "-----" << std::endl;

    std::cout << "large vector dotProduct float2" << std::endl;
    DotProductCuda<float2> dotProductFloat2Large(16777216);
    dotProductFloat2Large.Process(contextFloat2);
    std::cout << "-----" << std::endl;

    std::cout << "large vector dotProduct float4" << std::endl;
    DotProductCuda<float4> dotProductFloat4Large(16777216);
    dotProductFloat4Large.Process(contextFloat4);
    std::cout << "-----" << std::endl;

    std::cout << "small vector dotProduct float (A = 1.0 and B = 1.0)" << std::endl;
    DotProductCuda<float> dotProductFloatSmall(128);
    dotProductFloatSmall.Process(contextFloat);
    std::cout << "-----" << std::endl;

    std::cout << "small vector dotProduct float2" << std::endl;
    DotProductCuda<float2> dotProductFloat2Small(128);
    dotProductFloat2Small.Process(contextFloat2);
    std::cout << "-----" << std::endl;

    std::cout << "small vector dotProduct float4" << std::endl;
    DotProductCuda<float4> dotProductFloat4Small(128);
    dotProductFloat4Small.Process(contextFloat4);
    std::cout << "-----" << std::endl;
}

void dotProductOpenCL()
{
    std::cout << std::endl << "[OpenCL] DotProduct tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "dotProduct.cl";

    openclModule.Compile(kernelFile);

    DotProductOpenCL<float> dotProductFloatOperation(16777216);
    OpenCLContext context = openclModule.GetContext("dotProductFloat");
    dotProductFloatOperation.Process(context);
    std::cout << "OpenCL test passed: dotProductFloat" << std::endl;

    DotProductOpenCL<cl_float2> dotProductFloat2Operation(16777216);
    context = openclModule.GetContext("dotProductFloat2");
    dotProductFloat2Operation.Process(context);
    std::cout << "OpenCL test passed: dotProductFloat2" << std::endl;

    DotProductOpenCL<cl_float4> dotProductFloat4Operation(16777216);
    context = openclModule.GetContext("dotProductFloat4");
    dotProductFloat4Operation.Process(context);
    std::cout << "OpenCL test passed: dotProductFloat4" << std::endl;
}

void dgemvCuda()
{
    std::cout << std::endl << "[Cuda] DGEMV tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "dgemv.cu";

    cudaModule.Compile(kernelFile);

    DgemvCuda dgemvOperation(4096, 4096*2);
    CudaContext context = cudaModule.GetContext("dgemv");
    dgemvOperation.Process(context);
    std::cout << "Cuda test passed: dgemv" << std::endl;
}

void dgemvOpenCL()
{
    std::cout << std::endl << "[OpenCL] DGEMV tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "dgemv.cl";

    openclModule.Compile(kernelFile);

    DgemvOpenCL dgemvOperation(4096, 4096 * 2);
    OpenCLContext context = openclModule.GetContext("dgemv");
    dgemvOperation.Process(context);
    std::cout << "OpenCL test passed: dgemv" << std::endl;
}

int main(int argc, char** argv)
{
    matAddCuda();
    dotProductCuda();
    dgemvCuda();

    matAddOpenCL();
    dgemvOpenCL();
    dotProductOpenCL();

    return 0;
}
