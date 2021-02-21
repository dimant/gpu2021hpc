#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

#include <cuda.h>
#include "cuda_util.h"
#include "util.h"

struct CudaContext
{
	CUfunction cuFunction;
	Work work;
};

class CudaModule
{
private:
	CUdevice cuDevice;
	CUcontext cuContext;
	CUlinkState cuLinkState;
	CUmodule cuModule;
	CUfunction cuFunction;

public:
	CudaModule() :
		cuDevice(),
		cuLinkState(nullptr),
		cuContext(nullptr),
		cuModule(nullptr),
		cuFunction(nullptr)
	{
	}

	~CudaModule()
	{
		checkCudaError(cuLinkDestroy(cuLinkState));
		checkCudaError(cuCtxDestroy(cuContext));
	}

	void Init();

	void Compile(const char* kernelFile);

	CudaContext GetContext(const char* kernelName);
};

#endif