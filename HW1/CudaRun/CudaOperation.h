#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <cuda.h>

class CudaOperation
{
private:

	CUfunction cuFunction;

public:
	CudaOperation() :
		cuFunction(nullptr)
	{

	}

	void SetFunction(CUfunction cuF)
	{
		cuFunction = cuF;
	}

	CUfunction GetFunction()
	{
		return cuFunction;
	}


	void Process(CUfunction cuFunction);

	virtual void AllocateHost() = 0;
	virtual void AllocateDevice() = 0;
	virtual void InitData() = 0;
	virtual void CopyToDevice() = 0;
	virtual void Launch() = 0;
	virtual void CopyFromDevice() = 0;
	virtual void VerifyResult() = 0;
	virtual void FreeHost() = 0;
	virtual void FreeDevice() = 0;
};

#endif