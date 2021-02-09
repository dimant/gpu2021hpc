#include <iostream>

#include "CudaModule.h"
#include "OpenCLModule.h"

#include "MatAddCuda.h"
#include "MatAddOpenCL.h"

#include "DotProductCuda.h"
#include "DotProductOpenCL.h"

#include "DgemvCuda.h"
#include "DgemvOpenCL.h"

#include "GemmCuda.h"
#include "GemmOpenCL.h"

#include "TransposeCuda.h"
#include "TransposeOpenCL.h"

#include "BlurCuda.h"
#include "BlurOpenCL.h"

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
    std::cout << "Cuda test completed: matAdd" << std::endl;

    context = cudaModule.GetContext("matAddRow");
    matAddOperation.Process(context);
    std::cout << "Cuda test completed: matAddRow" << std::endl;

    context = cudaModule.GetContext("matAddCol");
    matAddOperation.Process(context);
    std::cout << "Cuda test completed: matAddCol" << std::endl;
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
    std::cout << "OpenCL test completed: matAdd" << std::endl;

    context = openclModule.GetContext("matAddRow");
    matAddOperation.Process(context);
    std::cout << "OpenCL test completed: matAddRow" << std::endl;

    context = openclModule.GetContext("matAddCol");
    matAddOperation.Process(context);
    std::cout << "OpenCL test completed: matAddCol" << std::endl;
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
    std::cout << "OpenCL test completed: dotProductFloat" << std::endl;

    DotProductOpenCL<cl_float2> dotProductFloat2Operation(16777216);
    context = openclModule.GetContext("dotProductFloat2");
    dotProductFloat2Operation.Process(context);
    std::cout << "OpenCL test completed: dotProductFloat2" << std::endl;

    DotProductOpenCL<cl_float4> dotProductFloat4Operation(16777216);
    context = openclModule.GetContext("dotProductFloat4");
    dotProductFloat4Operation.Process(context);
    std::cout << "OpenCL test completed: dotProductFloat4" << std::endl;
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
    std::cout << "Cuda test completed: dgemv" << std::endl;
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
    std::cout << "OpenCL test completed: dgemv" << std::endl;
}

void gemmCuda()
{
    std::cout << std::endl << "[Cuda] GEMM tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "gemm.cu";

    cudaModule.Compile(kernelFile);

    GemmCuda<float> sgemmOperation(3, 3, 3, 3);
    CudaContext context = cudaModule.GetContext("sgemm");
    sgemmOperation.Process(context);
    std::cout << "Cuda test completed: sgemm" << std::endl;

    GemmCuda<double> dgemmOperation(3, 3, 3, 3);
    context = cudaModule.GetContext("dgemm");
    dgemmOperation.Process(context);
    std::cout << "Cuda test completed: dgemm" << std::endl;
}

void gemmOpenCL()
{
    std::cout << std::endl << "[OpenCL] GEMM tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "gemm.cl";

    openclModule.Compile(kernelFile);

    GemmOpenCL<float> sgemmOperation(3, 3, 3, 3);
    OpenCLContext context = openclModule.GetContext("sgemm");
    sgemmOperation.Process(context);
    std::cout << "OpenCL test completed: sgemm" << std::endl;

    GemmOpenCL<double> dgemmOperation(3, 3, 3, 3);
    context = openclModule.GetContext("dgemm");
    dgemmOperation.Process(context);
    std::cout << "OpenCL test completed: dgemm" << std::endl;
}

void transposeCuda()
{
    std::cout << std::endl << "[Cuda] transpose tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "transpose.cu";

    cudaModule.Compile(kernelFile);

    TransposeCuda transposeOperation(3, 3);
    CudaContext context = cudaModule.GetContext("transpose");
    transposeOperation.Process(context);

    std::cout << "Cuda test completed: transpose" << std::endl;
}

void transposeOpenCL()
{
    std::cout << std::endl << "[OpenCL] transpose tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "transpose.cl";

    openclModule.Compile(kernelFile);

    TransposeOpenCL transposeOperation(3, 3);
    OpenCLContext context = openclModule.GetContext("transpose");
    transposeOperation.Process(context);

    std::cout << "OpenCL test completed: transpose" << std::endl;
}

void blurCuda()
{
    std::cout << std::endl << "[Cuda] blur tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    const char* kernelFile = "blur.cu";

    cudaModule.Compile(kernelFile);

    std::cout << "Testing: blur3x3" << std::endl;
    BlurCuda blurOperation3x3(100, 100, 1);
    CudaContext context = cudaModule.GetContext("blur3x3");
    blurOperation3x3.Process(context);
    std::cout << "Cuda test completed: blur3x3" << std::endl << std::endl;

    std::cout << "Testing: blur9x9" << std::endl;
    BlurCuda blurOperation9x9(100, 100, 4);
    context = cudaModule.GetContext("blur9x9");
    blurOperation9x9.Process(context);
    std::cout << "Cuda test completed: blur9x9" << std::endl << std::endl;

    std::cout << "Testing: blurMxM" << std::endl;
    BlurCuda blurOperationMxM(100, 100, 7);
    context = cudaModule.GetContext("blurMxM");
    blurOperationMxM.Process(context);
    std::cout << "Cuda test completed: blurMxM" << std::endl << std::endl;
}

void blurOpenCL()
{
    std::cout << std::endl << "[OpenCL] blur tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "blur.cl";

    openclModule.Compile(kernelFile);

    std::cout << "Testing: blur3x3" << std::endl;
    BlurOpenCL blurOperation3x3(100, 100, 1);
    OpenCLContext context = openclModule.GetContext("blur3x3");
    blurOperation3x3.Process(context);
    std::cout << "OpenCL test completed: blur3x3" << std::endl << std::endl;

    std::cout << "Testing: blur9x9" << std::endl;
    BlurOpenCL blurOperation9x9(100, 100, 4);
    context = openclModule.GetContext("blur9x9");
    blurOperation9x9.Process(context);
    std::cout << "OpenCL test completed: blur9x9" << std::endl << std::endl;

    std::cout << "Testing: blurMxM" << std::endl;
    BlurOpenCL blurOperationMxM(100, 100, 7);
    context = openclModule.GetContext("blurMxM");
    blurOperationMxM.Process(context);
    std::cout << "OpenCL test completed: blurMxM" << std::endl << std::endl;
}

int main(int argc, char** argv)
{
    //matAddCuda();
    //matAddOpenCL();

    //dotProductCuda();
    //dotProductOpenCL();

    //dgemvCuda();
    //dgemvOpenCL();

    //gemmCuda();
    //gemmOpenCL();

    //transposeCuda();
    //transposeOpenCL();

    blurCuda();
    blurOpenCL();

    return 0;
}
