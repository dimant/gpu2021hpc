#ifndef TRANSPOSECUDA_H
#define TRANSPOSECUDA_H

#include <cuda.h>
#include "CudaModule.h"
#include "HpcOperation.h"
#include "TransposeOperation.h"

class TransposeCuda : public HpcOperation<CudaContext>, public TransposeOperation
{
private:
	CUdeviceptr d_A;
	CUdeviceptr d_B;

public:
	TransposeCuda(size_t rows, size_t cols) :
		d_A(0), d_B(0), TransposeOperation(rows, cols)
	{
	}

	void AllocateHost()
	{
		TransposeOperation::AllocateHost();
	}

	void InitData()
	{
		TransposeOperation::InitData();
	}

	void VerifyResult()
	{
		TransposeOperation::VerifyResult();
	}

	void FreeHost()
	{
		TransposeOperation::FreeHost();
	}

	void Launch();
	void CopyFromDevice();
	void CopyToDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif