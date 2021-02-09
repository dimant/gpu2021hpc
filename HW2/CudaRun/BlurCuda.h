#ifndef BLURCUDA_H
#define BLURCUDA_H

#include <cuda.h>
#include "CudaModule.h"
#include "HpcOperation.h"
#include "BlurOperation.h"

class BlurCuda : public HpcOperation<CudaContext>, public BlurOperation
{
private:
	CUdeviceptr d_A;
	CUdeviceptr d_B;

public:
	BlurCuda(size_t rows, size_t cols, int blurSize) :
		d_A(0), d_B(0), BlurOperation(rows, cols, blurSize)
	{
	}

	void AllocateHost()
	{
		BlurOperation::AllocateHost();
	}

	void InitData()
	{
		BlurOperation::InitData();
	}

	void VerifyResult()
	{
		BlurOperation::VerifyResult();
	}

	void FreeHost()
	{
		BlurOperation::FreeHost();
	}

	void Launch();
	void CopyFromDevice();
	void CopyToDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif