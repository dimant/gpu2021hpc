#ifndef GEMMCUDA_H
#define GEMMVCUDA_H

#include <cuda.h>

#include "CudaModule.h"
#include "HpcOperation.h"
#include "GemmOperation.h"

class GemmCuda : public HpcOperation<CudaContext>, public GemmOperation
{
private:
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;

public:
	GemmCuda(int widthA, int heightA, int widthB, int heightB) :
		d_A(0), d_B(0), d_C(0),
		GemmOperation(widthA, heightA, widthB, heightB)
	{
	}

	void AllocateHost()
	{
		GemmOperation::AllocateHost();
	}

	void InitData()
	{
		GemmOperation::InitData();
	}

	void VerifyResult()
	{
		GemmOperation::VerifyResult();
	}

	void FreeHost()
	{
		GemmOperation::FreeHost();
	}

	void Launch();
	void CopyToDevice();
	void CopyFromDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif