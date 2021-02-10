#include <iostream>

#include "popl.h"

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

void matAddCuda(int rowcols, int threads, int blocks, const char* kernelName)
{
    std::cout << std::endl << "[CUDA] MatAdd tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    cudaModule.Compile("matAdd.cu");

    MatAddCuda matAddOperation(rowcols, rowcols);
    CudaContext context = cudaModule.GetContext(kernelName);
    context.work.threads.x = threads;
    context.work.threads.y = threads;
    context.work.blocks.x = blocks;
    context.work.blocks.y = blocks;
    matAddOperation.Process(context);
    std::cout << "Cuda test completed: " << kernelName << std::endl;
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
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    matAddOperation.Process(context);
    std::cout << "OpenCL test completed: matAdd" << std::endl;

    context = openclModule.GetContext("matAddRow");
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
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

void dgemvCuda(int rowcols, int threads, int blocks)
{
    std::cout << std::endl << "[Cuda] DGEMV tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    cudaModule.Compile("dgemv.cu");

    DgemvCuda dgemvOperation(rowcols, rowcols);
    CudaContext context = cudaModule.GetContext("dgemv");
    context.work.threads.x = threads;
    context.work.threads.y = threads;
    context.work.blocks.x = blocks;
    context.work.blocks.y = blocks;
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
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    dgemvOperation.Process(context);
    std::cout << "OpenCL test completed: dgemv" << std::endl;
}

void gemmCuda(int rowcols, int threads, int blocks, int type)
{
    std::cout << std::endl << "[Cuda] GEMM tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    cudaModule.Compile("gemm.cu");

    if (type == 1)
    {
        GemmCuda<float> sgemmOperation(rowcols, rowcols, rowcols, rowcols);
        CudaContext context = cudaModule.GetContext("sgemm");
        context.work.threads.x = threads;
        context.work.threads.y = threads;
        context.work.blocks.x = blocks;
        context.work.blocks.y = blocks;
        sgemmOperation.Process(context);
        std::cout << "Cuda test completed: sgemm" << std::endl;
    }
    else
    {
        GemmCuda<double> sgemmOperation(rowcols, rowcols, rowcols, rowcols);
        CudaContext context = cudaModule.GetContext("dgemm");
        context.work.threads.x = threads;
        context.work.threads.y = threads;
        context.work.blocks.x = blocks;
        context.work.blocks.y = blocks;
        sgemmOperation.Process(context);
        std::cout << "Cuda test completed: dgemm" << std::endl;
    }

}

void gemmOpenCL()
{
    std::cout << std::endl << "[OpenCL] GEMM tests" << std::endl;

    OpenCLModule openclModule;
    openclModule.Init();

    const char* kernelFile = "gemm.cl";

    openclModule.Compile(kernelFile);

    GemmOpenCL<float> sgemmOperation(100, 100, 100, 100);
    OpenCLContext context = openclModule.GetContext("sgemm");
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    sgemmOperation.Process(context);
    std::cout << "OpenCL test completed: sgemm" << std::endl << std::endl;

    GemmOpenCL<double> dgemmOperation(100, 100, 100, 100);
    context = openclModule.GetContext("dgemm");
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    dgemmOperation.Process(context);
    std::cout << "OpenCL test completed: dgemm" << std::endl;
}

void transposeCuda(int rowcols, int threads, int blocks)
{
    std::cout << std::endl << "[Cuda] transpose tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    cudaModule.Compile("transpose.cu");

    TransposeCuda transposeOperation(rowcols, rowcols);
    CudaContext context = cudaModule.GetContext("transpose");
    context.work.threads.x = threads;
    context.work.threads.y = threads;
    context.work.blocks.x = blocks;
    context.work.blocks.y = blocks;
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

    TransposeOpenCL transposeOperation(100, 100);
    OpenCLContext context = openclModule.GetContext("transpose");
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    transposeOperation.Process(context);

    std::cout << "OpenCL test completed: transpose" << std::endl;
}


/// <summary>
/// blurSize: 1 for 3x3, 4 for 9x9 and anything else for MxM
/// </summary>
void blurCuda(int rowcols, int threads, int blocks, int blurSize)
{
    std::cout << std::endl << "[Cuda] blur tests" << std::endl;

    CudaModule cudaModule;
    cudaModule.Init();

    cudaModule.Compile("blur.cu");

    std::string kernelName;

    if (blurSize == 1)
    {
        kernelName = "blur3x3";
    }
    else if (blurSize == 4)
    {
        kernelName = "blur9x9";
    }
    else
    {
        kernelName = "blurMxM";
    }

    std::cout << "Testing: " << kernelName << std::endl;
    BlurCuda blurOperation3x3(rowcols, rowcols, 1);
    CudaContext context = cudaModule.GetContext(kernelName.c_str());
    context.work.threads.x = threads;
    context.work.threads.y = threads;
    context.work.blocks.x = blocks;
    context.work.blocks.y = blocks;
    blurOperation3x3.Process(context);
    std::cout << "Cuda test completed: " << kernelName << std::endl << std::endl;
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
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    blurOperation3x3.Process(context);
    std::cout << "OpenCL test completed: blur3x3" << std::endl << std::endl;

    std::cout << "Testing: blur9x9" << std::endl;
    BlurOpenCL blurOperation9x9(100, 100, 4);
    context = openclModule.GetContext("blur9x9");
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    blurOperation9x9.Process(context);
    std::cout << "OpenCL test completed: blur9x9" << std::endl << std::endl;

    std::cout << "Testing: blurMxM" << std::endl;
    BlurOpenCL blurOperationMxM(100, 100, 7);
    context = openclModule.GetContext("blurMxM");
    context.work.threads.x = 32;
    context.work.threads.y = 32;
    context.work.blocks.x = 4;
    context.work.blocks.y = 4;
    blurOperationMxM.Process(context);
    std::cout << "OpenCL test completed: blurMxM" << std::endl << std::endl;
}

int main(int argc, char** argv)
{
    popl::OptionParser op("Allowed options");

    int threads;
    int blocks;
    int rowcols;
    int blurSize;

    auto help_option = op.add<popl::Switch>("h", "help", "produce help message");
    auto threads_option = op.add<popl::Value<int>>("t", "threads", "Number of x and y threads", 32, &threads);
    auto blocks_option = op.add<popl::Value<int>>("b", "blocks", "Number of x and y blocks", 4, &blocks);
    auto rowcols_option = op.add<popl::Value<int>>("r", "rows-cols", "Number of matrix rows and cols", 128, &rowcols);
    auto blurSize_option = op.add<popl::Value<int>>("s", "blur-size", "1/2 of blur size", 1, &blurSize);
    auto kernelName_option = op.add<popl::Value<std::string>>("k", "kernel-name", "Name of kernel.");

    op.parse(argc, argv);

    if (!kernelName_option->is_set() || help_option->is_set())
    {
        std::cout << op << "\n";
    }
    else
    {
        std::string kernelName = kernelName_option->value();

        if (kernelName.rfind("matAdd", 0) == 0)
        {
            matAddCuda(rowcols, threads, blocks, kernelName.c_str());
        }
        if (kernelName.rfind("dgemv", 0) == 0)
        {
            dgemvCuda(rowcols, threads, blocks);
        }
        else if (kernelName.rfind("sgemm", 0) == 0)
        {
            gemmCuda(rowcols, threads, blocks, 1);
        }
        else if (kernelName.rfind("dgemm", 0) == 0)
        {
            gemmCuda(rowcols, threads, blocks, 2);
        }
        else if(kernelName.rfind("transpose", 0) == 0)
        {
            transposeCuda(rowcols, threads, blocks);
        }
        else if (kernelName.rfind("blur", 0) == 0)
        {
            if (!blurSize_option->is_set())
            {
                std::cout << op << "\n";
            }
            else
            {
                blurCuda(rowcols, threads, blocks, blurSize);
            }
        }
    }

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

    //blurCuda();
    //blurOpenCL();

    return 0;
}
