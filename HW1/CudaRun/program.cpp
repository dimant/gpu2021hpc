#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <device_launch_parameters.h>

#include <cuda.h>
#include "cuda_util.h"

#include "CudaModule.h"
#include "MatAddOperation.h"

int main(int argc, char** argv)
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

    return 0;
}
